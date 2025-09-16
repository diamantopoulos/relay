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
import os

from .graph import CSRGraph
from .kernels import (
    c2v_min_sum_kernel, v2c_and_marginals_kernel, v2c_and_marginals_fused_gamma_kernel,
    gamma_mix_kernel, parity_per_check_kernel, init_messages_kernel, stop_flag_kernel, get_autotuner,
    be_to_eb_kernel, eb_to_be_kernel, c2v_min_sum_btile_kernel, v2c_and_gamma_btile_kernel,
    relay_decode_persistent_kernel,
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
        dtype_messages: str = "fp16",  # Default to fp16 for better performance
        device: str = "cuda",
        seed: int = 1234,
        bitpack_output: bool = False,
        mode: str = "default",  # "default" or "throughput"
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
        self.mode = mode
        
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
        
        # Set up message dtype (fp16 for messages, fp32 for accumulations)
        if dtype_messages == "fp16":
            self.msg_dtype = torch.float16
            self.acc_dtype = torch.float32  # Accumulations in fp32
        elif dtype_messages == "fp32":
            self.msg_dtype = torch.float32
            self.acc_dtype = torch.float32
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
        
        # Constants for device-driven early exit
        self.CHECK_EVERY = 16  # Check stop flag every N iterations
        
        # Constants for kernel batching (reduce launch overhead)
        self.ROWS_PER_PROG_C2V = 8  # Number of rows per C2V program
        self.ROWS_PER_PROG_V2C = 8  # Number of rows per V2C program
        self.ROWS_PER_PROG_PAR = 8  # Number of rows per parity program

        # Realtime env toggles
        self.rt_Q = int(os.getenv("RELAY_RT_QUEUE", "64"))
        self.rt_check_every = int(os.getenv("RELAY_RT_CHECK_EVERY", "8"))
        self.rt_msg_dtype = os.getenv("RELAY_RT_MSG_DTYPE", "fp16")
        self.rt_num_warps = int(os.getenv("RELAY_RT_PERSIST_WARPS", "8"))
        self.rt_block_size = int(os.getenv("RELAY_RT_BLOCK_SIZE", "128"))
        self._rt_worker_launched = False
    
    def _allocate_device_tensors(self, B: int):
        """Allocate device tensors for batch size B."""
        if B in self._device_tensors:
            return self._device_tensors[B]
        
        tensors = {
            # Messages (fp16 for storage, fp32 for accumulations)
            'mu': torch.zeros(B, self.E, dtype=self.msg_dtype, device=self.device),
            'nu': torch.zeros(B, self.E, dtype=self.msg_dtype, device=self.device),
            
            # Beliefs (fp32 for accuracy)
            'lambda_': torch.zeros(B, self.V, dtype=self.acc_dtype, device=self.device),
            'M': torch.zeros(B, self.V, dtype=self.acc_dtype, device=self.device),
            'hard_dec': torch.zeros(B, self.V, dtype=torch.uint8, device=self.device),
            
            # Gamma (fp32 for accuracy)
            'gamma': torch.zeros(B, self.V, dtype=self.acc_dtype, device=self.device),
            
            # Syndrome and validation
            'syndrome': torch.zeros(B, self.C, dtype=torch.uint8, device=self.device),
            'check_ok': torch.zeros(B, self.C, dtype=torch.uint8, device=self.device),
            
            # Solution tracking
            'best_weights': torch.full((B,), float('inf'), dtype=torch.float32, device=self.device),
            'best_errors': torch.zeros(B, self.V, dtype=torch.uint8, device=self.device),
            'found_count': torch.zeros(B, dtype=torch.int32, device=self.device),
            'valid_solutions': torch.zeros(B, dtype=torch.bool, device=self.device),
            
            # Device-driven early exit
            'stop_flag': torch.zeros(1, dtype=torch.uint8, device=self.device),
        }
        if self.mode == "throughput":
            # [E,B] transposed message buffers for BTILE kernels
            tensors['muT'] = torch.zeros(self.E, B, dtype=self.msg_dtype, device=self.device)
            tensors['nuT'] = torch.zeros(self.E, B, dtype=self.msg_dtype, device=self.device)
        self._device_tensors[B] = tensors
        return tensors

    def _allocate_realtime_state(self):
        if hasattr(self, "_rt_state_allocated") and self._rt_state_allocated:
            return
        Q = self.rt_Q
        # queues
        self.rt_syndromes = torch.zeros(Q, self.C, dtype=torch.uint8, device=self.device)
        self.rt_errors = torch.zeros(Q, self.V, dtype=torch.uint8, device=self.device)
        self.rt_weights = torch.zeros(Q, dtype=torch.float32, device=self.device)
        self.rt_valid = torch.zeros(Q, dtype=torch.uint8, device=self.device)
        self.rt_slot_state = torch.zeros(Q, dtype=torch.int32, device=self.device)  # 0=EMPTY,1=READY,2=DONE
        # state B=1
        msg_dtype = torch.float16 if self.rt_msg_dtype == "fp16" else torch.float32
        self.rt_mu = torch.zeros(self.E, dtype=msg_dtype, device=self.device)
        self.rt_nu = torch.zeros(self.E, dtype=msg_dtype, device=self.device)
        self.rt_lambda = torch.zeros(self.V, dtype=torch.float32, device=self.device)
        self.rt_M = torch.zeros(self.V, dtype=torch.float32, device=self.device)
        self.rt_gamma = torch.zeros(self.V, dtype=torch.float32, device=self.device)
        self.rt_hard_dec = torch.zeros(self.V, dtype=torch.uint8, device=self.device)
        self.rt_check_ok = torch.zeros(self.C, dtype=torch.uint8, device=self.device)
        self._rt_head = 0
        self._rt_state_allocated = True

    def _ensure_worker(self):
        if self._rt_worker_launched:
            return
        self._allocate_realtime_state()
        # launch persistent worker
        grid = (max(self.C // 16, 1),)
        msg_is_fp16 = (self.rt_mu.dtype == torch.float16)
        relay_decode_persistent_kernel[grid](
            self.graph.chk_ptr, self.graph.chk_edges, self.graph.var_ptr, self.graph.var_edges, self.graph.edge_var, self.graph.edge_chk,
            self.rt_mu, self.rt_nu, self.rt_lambda, self.rt_M, self.rt_gamma, self.rt_hard_dec, self.rt_check_ok,
            self.lambda0,
            self.rt_syndromes, self.rt_errors, self.rt_weights, self.rt_valid, self.rt_slot_state,
            self.rt_Q, self.C, self.V, self.E,
            self.normalized_min_sum_alpha or 0.0, self.offset_min_sum_beta or 0.0,
            self.pre_iter, self.set_max_iter, self.num_sets, self.rt_check_every,
            use_alpha=(self.normalized_min_sum_alpha is not None),
            use_beta=(self.offset_min_sum_beta is not None),
            msg_is_fp16=msg_is_fp16,
            BLOCK_SIZE=self.rt_block_size,
            ROWS_PER_CHK=8,
            ROWS_PER_VAR=8,
            MAX_STEPS=1000000,
            num_warps=self.rt_num_warps,
            num_stages=2,
            )
        self._rt_worker_launched = True

    def decode_rt(self, syndrome: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Realtime single-sample decode using persistent worker.
        syndrome: [1, C] uint8 on CUDA device
        """
        assert self.mode == "realtime", "Use mode='realtime' to call decode_rt()"
        if syndrome.device != torch.device(self.device):
            syndrome = syndrome.to(self.device)
        if syndrome.ndim == 1:
            syndrome = syndrome.view(1, -1)
        assert syndrome.shape == (1, self.C)
        self._ensure_worker()
        Q = self.rt_Q
        slot = self._rt_head % Q
        # ensure slot empty
        # write syndrome
        self.rt_syndromes[slot].copy_(syndrome[0])
        # set READY
        self.rt_slot_state[slot] = 1
        # wait busy-loop (simple polling)
        # WARNING: simple implementation; can be replaced by event-based
        while True:
            state = int(self.rt_slot_state[slot].item())
            if state == 2:
                break
            torch.cuda._sleep(1000)  # ~1us
        # read outputs
        errors = self.rt_errors[slot:slot+1]
        weights = self.rt_weights[slot:slot+1]
        valid = self.rt_valid[slot:slot+1].to(torch.bool)
        # reset slot
        self.rt_slot_state[slot] = 0
        self._rt_head += 1
        return {"errors": errors, "weights": weights, "valid_mask": valid}
    
    # --- Transpose helpers ---
    def _be_to_eb(self, src_be: torch.Tensor, dst_eb: torch.Tensor, B: int):
        grid = (self.E,)
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        be_to_eb_kernel[grid](
            src_be, dst_eb, B, self.E,
            msg_is_fp16, BTILE=16,
        )

    def _eb_to_be(self, src_eb: torch.Tensor, dst_be: torch.Tensor, B: int):
        grid = (self.E,)
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        eb_to_be_kernel[grid](
            src_eb, dst_be, B, self.E,
            msg_is_fp16, BTILE=16,
        )

    def _launch_c2v_btile(self, tensors: Dict, B: int):
        grid = (self.C, (B + 16 - 1) // 16)
        alpha = self.normalized_min_sum_alpha or 0.0
        beta = self.offset_min_sum_beta or 0.0
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        c2v_min_sum_btile_kernel[grid](
            tensors['muT'], tensors['nuT'],
            self.graph.chk_ptr, self.graph.chk_edges,
            tensors['syndrome'], B, self.C, self.E,
            alpha, beta, use_alpha, use_beta,
            msg_is_fp16,
            self.c2v_config['BLOCK_SIZE'],
            BTILE=16,
            num_warps=self.c2v_config['num_warps'],
            num_stages=self.c2v_config['num_stages'],
        )
    
    def _launch_c2v_kernel(self, tensors: Dict, B: int):
        """Launch check-to-variable kernel."""
        alpha = self.normalized_min_sum_alpha or 0.0
        beta = self.offset_min_sum_beta or 0.0
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None
        
        total = B * self.C
        grid = ((total + self.ROWS_PER_PROG_C2V - 1) // self.ROWS_PER_PROG_C2V,)
        
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        
        c2v_min_sum_kernel[grid](
            tensors['mu'], tensors['nu'],
            self.graph.chk_ptr, self.graph.chk_edges,
            tensors['syndrome'],                      # pass syndrome here
            B, self.C, self.E,
            alpha, beta, use_alpha, use_beta,
            msg_is_fp16,
            self.c2v_config['BLOCK_SIZE'],
            ROWS_PER_PROG=self.ROWS_PER_PROG_C2V,
            num_warps=self.c2v_config['num_warps'],
            num_stages=self.c2v_config['num_stages']
        )
    
    def _launch_v2c_kernel(self, tensors: Dict, B: int):
        """Launch variable-to-check and marginals kernel."""
        grid = (B * self.V,)
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        
        v2c_and_marginals_kernel[grid](
            tensors['nu'], tensors['mu'],
            self.graph.var_ptr, self.graph.var_edges,
            tensors['lambda_'], tensors['M'], tensors['hard_dec'],
            B, self.V, self.E,
            msg_is_fp16,
            self.v2c_config['BLOCK_SIZE'],
            num_warps=self.v2c_config['num_warps'],
            num_stages=self.v2c_config['num_stages']
        )
    
    def _launch_v2c_fused_gamma_kernel(self, tensors: Dict, B: int):
        """Launch variable-to-check and marginals kernel with fused gamma mixing."""
        total = B * self.V
        grid = ((total + self.ROWS_PER_PROG_V2C - 1) // self.ROWS_PER_PROG_V2C,)
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        
        v2c_and_marginals_fused_gamma_kernel[grid](
            tensors['nu'], tensors['mu'],
            self.graph.var_ptr, self.graph.var_edges,
            tensors['lambda_'], tensors['M'], tensors['hard_dec'],
            self.lambda0, tensors['gamma'],
            B, self.V, self.E,
            msg_is_fp16,
            self.v2c_config['BLOCK_SIZE'],
            STORE_M=False,  # Don't spill M each iter for bandwidth optimization
            ROWS_PER_PROG=self.ROWS_PER_PROG_V2C,
            num_warps=self.v2c_config['num_warps'],
            num_stages=self.v2c_config['num_stages']
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
        total = B * self.C
        grid = ((total + self.ROWS_PER_PROG_PAR - 1) // self.ROWS_PER_PROG_PAR,)
        
        parity_per_check_kernel[grid](
            tensors['hard_dec'],
            self.graph.chk_ptr, self.graph.chk_edges, self.graph.edge_var,
            tensors['syndrome'], tensors['check_ok'],
            B, self.C, self.V, self.E,
            self.c2v_config['BLOCK_SIZE'],
            ROWS_PER_PROG=self.ROWS_PER_PROG_PAR,
            num_warps=self.c2v_config['num_warps'],
            num_stages=self.c2v_config['num_stages'],
        )
    
    def _launch_init_kernel(self, tensors: Dict, B: int):
        """Launch message initialization kernel."""
        grid = (B * self.E,)
        
        init_messages_kernel[grid](
            tensors['mu'], tensors['nu'],
            B, self.E
        )
    
    def _launch_stop_flag_kernel(self, tensors: Dict, B: int):
        """Launch stop flag kernel."""
        grid = (1,)  # Single program
        
        stop_flag_kernel[grid](
            tensors['found_count'],
            self.stop_nconv,
            tensors['stop_flag'],
            B
        )
    
    def _launch_v2c_btile(self, tensors: Dict, B: int, write_hard: bool):
        grid = (self.V, (B + 16 - 1) // 16)
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        v2c_and_gamma_btile_kernel[grid](
            tensors['nuT'], tensors['muT'],
            self.graph.var_ptr, self.graph.var_edges,
            tensors['lambda_'], self.lambda0, tensors['gamma'],
            tensors['M'], tensors['hard_dec'],
            B, self.V, self.E,
            msg_is_fp16,
            self.v2c_config['BLOCK_SIZE'],
            BTILE=16,
            WRITE_HARD=write_hard,
            num_warps=self.v2c_config['num_warps'],
            num_stages=self.v2c_config['num_stages'],
        )
    
    def _check_parity_and_select(self, tensors: Dict, B: int):
        """Check parity and select best solutions (device-only, no return)."""
        # Launch parity check
        self._launch_parity_check_kernel(tensors, B)
        
        # Check if all parity constraints are satisfied
        valid = tensors['check_ok'].view(B, self.C).all(dim=1)  # [B] bool
        idx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # [K]
        
        if idx.numel() == 0:
            return  # No valid solutions, skip weight computation
        
        # Only compute weights for valid indices
        cand_w = (tensors['hard_dec'][idx].float() * self.wj).sum(dim=1)  # [K]
        
        better = cand_w < tensors['best_weights'][idx]
        if better.any():
            upd = idx[better]
            tensors['best_weights'][upd] = cand_w[better]
            tensors['best_errors'][upd] = tensors['hard_dec'][upd]
            tensors['valid_solutions'][upd] = True
        
        tensors['found_count'] += valid.to(torch.int32)
    
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
        
        if self.mode == "throughput":
            # transpose mu/nu to [E,B] once
            self._be_to_eb(tensors['mu'], tensors['muT'], B)
            self._be_to_eb(tensors['nu'], tensors['nuT'], B)

        for t in range(self.pre_iter):
            if self.mode == "throughput":
                # BTILE C2V and V2C fully in [E,B]
                self._launch_c2v_btile(tensors, B)
                self._launch_v2c_btile(tensors, B, write_hard=((t+1) % self.CHECK_EVERY == 0))
            else:
                self._launch_c2v_kernel(tensors, B)
                self._launch_v2c_fused_gamma_kernel(tensors, B)

            if (t + 1) % self.CHECK_EVERY == 0:
                # For parity path, we need hard_dec updated (done in v2c btile when write_hard=True)
                self._check_parity_and_select(tensors, B)
                self._launch_stop_flag_kernel(tensors, B)
                if tensors['stop_flag'].item():
                    break
        
        # Final parity check after pre-iterations
        self._check_parity_and_select(tensors, B)
        
        # Relay legs (R times)
        for leg in range(self.num_sets):
            # Sample disordered gamma
            gamma_min, gamma_max = self.gamma_dist_interval
            gamma_leg = sample_gamma_uniform(B, self.V, gamma_min, gamma_max, self.device, self.seed + leg)
            tensors['gamma'][:] = gamma_leg
            
            for t in range(self.set_max_iter):
                self._launch_c2v_kernel(tensors, B)                  # produces ν
                self._launch_v2c_fused_gamma_kernel(tensors, B)      # V2C + gamma mixing (no atomics)
                
                # Check parity and select solutions only every CHECK_EVERY iterations
                if (t + 1) % self.CHECK_EVERY == 0:
                    self._check_parity_and_select(tensors, B)
                    self._launch_stop_flag_kernel(tensors, B)
                    if tensors['stop_flag'].item():  # One tiny sync occasionally
                        break
            
            # Final parity check and stop flag after each leg
            self._check_parity_and_select(tensors, B)
            self._launch_stop_flag_kernel(tensors, B)
            if tensors['stop_flag'].item():  # One tiny sync occasionally
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
