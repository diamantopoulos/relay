# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Relay-BP-S decoder with Triton GPU kernels."""

import torch
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Optional, Tuple, Union

from .graph import CSRGraph
from .kernels import (
    c2v_min_sum_kernel, v2c_and_marginals_kernel, gamma_mix_kernel,
    parity_per_check_kernel, init_messages_kernel, get_autotuner
)
from .utils import (
    compute_log_prior_ratios, compute_decoding_weights, sample_gamma_uniform,
    sample_gamma_scalar, validate_error_priors, validate_csr_matrix,
    bitpack_errors, bitunpack_errors
)


class RelayBPDecoder:
    """Relay-BP-S decoder with Triton GPU kernels.
    
    Implements the Relay-BP algorithm with disordered memory strengths
    and ensemble decoding for quantum error correction.
    """
    
    def __init__(
        self,
        H_csr: csr_matrix,
        error_priors: np.ndarray,
        pre_iter: int = 80,
        num_sets: int = 100,
        set_max_iter: int = 60,
        gamma0: float = 0.65,
        gamma_dist_interval: Tuple[float, float] = (-0.24, 0.66),
        stop_nconv: int = 5,
        normalized_min_sum_alpha: Optional[float] = 0.90,
        offset_min_sum_beta: Optional[float] = None,
        dtype_messages: str = "fp16",
        device: str = "cuda",
        seed: int = 1234,
        bitpack_output: bool = False
    ):
        """Initialize Relay-BP decoder.
        
        Args:
            H_csr: Check matrix (C x V) in CSR format
            error_priors: [V] error probabilities in (0, 0.5)
            pre_iter: T0 - iterations for first set
            num_sets: R - number of relay legs/ensembles
            set_max_iter: Tr - iterations per relay set
            gamma0: γ for first set (ordered memory)
            gamma_dist_interval: (min, max) for disordered γ sampling
            stop_nconv: S - number of solutions to collect
            normalized_min_sum_alpha: α for normalized min-sum (0 < α ≤ 1)
            offset_min_sum_beta: β for offset min-sum (mutually exclusive with α)
            dtype_messages: "fp16" or "fp32" for message precision
            device: "cuda" or "rocm"
            seed: RNG seed for γ sampling
            bitpack_output: whether to return packed error bits
        """
        # Validate inputs
        validate_csr_matrix(H_csr)
        error_priors = validate_error_priors(error_priors)
        
        if normalized_min_sum_alpha is not None and offset_min_sum_beta is not None:
            raise ValueError("Cannot specify both normalized_min_sum_alpha and offset_min_sum_beta")
        
        if normalized_min_sum_alpha is not None and not (0 < normalized_min_sum_alpha <= 1):
            raise ValueError("normalized_min_sum_alpha must be in (0, 1]")
        
        if offset_min_sum_beta is not None and offset_min_sum_beta < 0:
            raise ValueError("offset_min_sum_beta must be >= 0")
        
        # Store parameters
        self.H_csr = H_csr
        self.error_priors = error_priors
        self.pre_iter = pre_iter
        self.num_sets = num_sets
        self.set_max_iter = set_max_iter
        self.gamma0 = gamma0
        self.gamma_dist_interval = gamma_dist_interval
        self.stop_nconv = stop_nconv
        self.normalized_min_sum_alpha = normalized_min_sum_alpha
        self.offset_min_sum_beta = offset_min_sum_beta
        self.dtype_messages = dtype_messages
        self.device = device
        self.seed = seed
        self.bitpack_output = bitpack_output
        
        # Set up device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        if device == "cpu":
            raise RuntimeError("CPU not supported - Triton kernels require GPU. Use device='cuda' or device='rocm'")
        
        # Build graph representation
        self.graph = CSRGraph(H_csr, device)
        self.C, self.V = self.graph.C, self.graph.V
        self.E = self.graph.E
        
        # Compute log prior ratios
        log_ratios = compute_log_prior_ratios(error_priors)
        self.lambda0 = torch.from_numpy(log_ratios).to(device).unsqueeze(0)  # [1, V]
        
        # Precompute weights once (avoid recomputing each loop)
        self.wj = torch.log(
            (1.0 - torch.from_numpy(self.error_priors).to(self.device, dtype=torch.float32))
            / torch.from_numpy(self.error_priors).to(self.device, dtype=torch.float32)
        ).contiguous()  # [V], device
        
        # Set up message dtype
        if dtype_messages == "fp16":
            self.msg_dtype = torch.float16
        elif dtype_messages == "fp32":
            self.msg_dtype = torch.float32
        else:
            raise ValueError("dtype_messages must be 'fp16' or 'fp32'")
        
        # Get autotuner
        self.autotuner = get_autotuner()
        
        # Get kernel configurations
        self.c2v_config = self.autotuner.tune_c2v_kernel(
            self.graph.max_deg_chk, self.msg_dtype, device
        )
        self.v2c_config = self.autotuner.tune_v2c_kernel(
            self.graph.max_deg_var, self.msg_dtype, device
        )
        
        # Set up RNG
        torch.manual_seed(seed)
        
        # Pre-allocate device tensors (will be allocated in decode)
        self._device_tensors = {}
    
    def _allocate_device_tensors(self, B: int):
        """Allocate device tensors for batch size B."""
        if B in self._device_tensors:
            return self._device_tensors[B]
        
        tensors = {
            # Messages
            'mu': torch.zeros(B, self.E, dtype=self.msg_dtype, device=self.device),
            'nu': torch.zeros(B, self.E, dtype=self.msg_dtype, device=self.device),
            
            # Beliefs
            'lambda_': torch.zeros(B, self.V, dtype=torch.float32, device=self.device),
            'M': torch.zeros(B, self.V, dtype=torch.float32, device=self.device),
            'hard_dec': torch.zeros(B, self.V, dtype=torch.uint8, device=self.device),
            
            # Gamma
            'gamma': torch.zeros(B, self.V, dtype=torch.float32, device=self.device),
            
            # Syndrome and validation
            'syndrome': torch.zeros(B, self.C, dtype=torch.uint8, device=self.device),
            'check_ok': torch.zeros(B, self.C, dtype=torch.uint8, device=self.device),
            
            # Solution tracking
            'best_weights': torch.full((B,), float('inf'), dtype=torch.float32, device=self.device),
            'best_errors': torch.zeros(B, self.V, dtype=torch.uint8, device=self.device),
            'found_count': torch.zeros(B, dtype=torch.int32, device=self.device),
            'valid_solutions': torch.zeros(B, dtype=torch.bool, device=self.device),
        }
        
        self._device_tensors[B] = tensors
        return tensors
    
    def _launch_c2v_kernel(self, tensors: Dict, B: int):
        """Launch check-to-variable kernel."""
        alpha = self.normalized_min_sum_alpha or 0.0
        beta = self.offset_min_sum_beta or 0.0
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None
        
        grid = (B * self.C,)
        
        c2v_min_sum_kernel[grid](
            tensors['mu'], tensors['nu'],
            self.graph.chk_ptr, self.graph.chk_edges,
            tensors['syndrome'],                      # pass syndrome here
            B, self.C, self.E,
            alpha, beta, use_alpha, use_beta,
            self.c2v_config['BLOCK_SIZE'],
            num_warps=self.c2v_config['num_warps']
        )
    
    def _launch_v2c_kernel(self, tensors: Dict, B: int):
        """Launch variable-to-check and marginals kernel."""
        grid = (B * self.V,)
        
        v2c_and_marginals_kernel[grid](
            tensors['nu'], tensors['mu'],
            self.graph.var_ptr, self.graph.var_edges,
            tensors['lambda_'], tensors['M'], tensors['hard_dec'],
            B, self.V, self.E,
            self.v2c_config['BLOCK_SIZE'],
            num_warps=self.v2c_config['num_warps']
        )
    
    def _launch_gamma_mix_kernel(self, tensors: Dict, B: int):
        """Launch gamma mixing kernel."""
        grid = (B * self.V,)
        
        gamma_mix_kernel[grid](
            self.lambda0, tensors['M'], tensors['gamma'],
            tensors['lambda_'],
            B, self.V
        )
    
    def _launch_parity_check_kernel(self, tensors: Dict, B: int):
        """Launch parity check kernel."""
        grid = (B * self.C,)
        
        parity_per_check_kernel[grid](
            tensors['hard_dec'],
            self.graph.chk_ptr, self.graph.chk_edges, self.graph.edge_var,
            tensors['syndrome'], tensors['check_ok'],
            B, self.C, self.V, self.E,
            self.c2v_config['BLOCK_SIZE'],
            num_warps=self.c2v_config['num_warps'],
        )
    
    def _launch_init_kernel(self, tensors: Dict, B: int):
        """Launch message initialization kernel."""
        grid = (B * self.E,)
        
        init_messages_kernel[grid](
            tensors['mu'], tensors['nu'],
            B, self.E
        )
    
    def _check_parity_and_select(self, tensors: Dict, B: int) -> torch.Tensor:
        """Check parity and select best solutions."""
        # Launch parity check
        self._launch_parity_check_kernel(tensors, B)
        
        # Check if all parity constraints are satisfied
        valid_per_check = tensors['check_ok'].view(B, self.C)  # uint8
        valid = torch.all(valid_per_check.bool(), dim=1)       # bool [B]
        cand_w = (tensors['hard_dec'].float() * self.wj).sum(dim=1)  # [B]

        better = valid & (cand_w < tensors['best_weights'])
        tensors['best_weights'] = torch.where(better, cand_w, tensors['best_weights'])
        tensors['best_errors'][better] = tensors['hard_dec'][better]
        tensors['valid_solutions'] |= better
        tensors['found_count'] += valid.to(torch.int32)
        
        return valid
    
    def decode(self, syndromes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode syndromes using Relay-BP algorithm.
        
        Args:
            syndromes: [B, C] syndrome bits (uint8)
            
        Returns:
            Dictionary with keys:
                - "errors": [B, V] decoded errors (uint8) or packed if bitpack_output=True
                - "weights": [B] solution weights (float32)
                - "valid_mask": [B] valid solution mask (bool)
        """
        B = syndromes.shape[0]
        
        if syndromes.shape[1] != self.C:
            raise ValueError(f"Expected {self.C} syndrome bits, got {syndromes.shape[1]}")
        
        if syndromes.device != torch.device(self.device):
            syndromes = syndromes.to(self.device)
        
        # Allocate device tensors
        tensors = self._allocate_device_tensors(B)
        
        # Copy syndromes to device
        tensors['syndrome'][:] = syndromes
        
        # Initialize
        tensors['lambda_'][:] = self.lambda0  # Broadcast prior LLRs
        tensors['best_weights'].fill_(float('inf'))
        tensors['found_count'].zero_()
        tensors['valid_solutions'].zero_()
        
        # Initialize messages
        self._launch_init_kernel(tensors, B)
        
        # Pre-iterations (T0) with uniform gamma0
        gamma0_tensor = sample_gamma_scalar(B, self.V, self.gamma0, self.device)
        tensors['gamma'][:] = gamma0_tensor
        
        for _ in range(self.pre_iter):
            self._launch_c2v_kernel(tensors, B)
            self._launch_v2c_kernel(tensors, B)
            self._launch_gamma_mix_kernel(tensors, B)
            
            # Check parity and select solutions
            valid = self._check_parity_and_select(tensors, B)
            
            # Early exit if all samples have enough solutions
            if torch.all(tensors['found_count'] >= self.stop_nconv):
                break
        
        # Relay legs (R times)
        for leg in range(self.num_sets):
            # Sample disordered gamma
            gamma_min, gamma_max = self.gamma_dist_interval
            gamma_leg = sample_gamma_uniform(B, self.V, gamma_min, gamma_max, self.device, self.seed + leg)
            tensors['gamma'][:] = gamma_leg
            
            for _ in range(self.set_max_iter):
                self._launch_c2v_kernel(tensors, B)
                self._launch_v2c_kernel(tensors, B)
                self._launch_gamma_mix_kernel(tensors, B)
                
                # Check parity and select solutions
                valid = self._check_parity_and_select(tensors, B)
                
                # Early exit if all samples have enough solutions
                if torch.all(tensors['found_count'] >= self.stop_nconv):
                    break
            
            # Early exit if all samples have enough solutions
            if torch.all(tensors['found_count'] >= self.stop_nconv):
                break
        
        # Prepare output
        if self.bitpack_output:
            errors = bitpack_errors(tensors['best_errors'])
        else:
            errors = tensors['best_errors']
        
        return {
            "errors": errors,
            "weights": tensors['best_weights'],
            "valid_mask": tensors['valid_solutions']
        }
    
    def get_stats(self) -> Dict:
        """Get decoder statistics."""
        return {
            "graph": self.graph.get_stats(),
            "parameters": {
                "pre_iter": self.pre_iter,
                "num_sets": self.num_sets,
                "set_max_iter": self.set_max_iter,
                "gamma0": self.gamma0,
                "gamma_dist_interval": self.gamma_dist_interval,
                "stop_nconv": self.stop_nconv,
                "dtype_messages": self.dtype_messages,
                "device": self.device,
            },
            "kernel_configs": {
                "c2v": self.c2v_config,
                "v2c": self.v2c_config,
            }
        }
