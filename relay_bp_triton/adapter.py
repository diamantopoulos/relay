"""
Triton adapter for Relay-BP decoder providing compatibility with relay_bp interface.

This module provides adapter classes that wrap the Triton GPU implementation
to match the interface expected by the relay_bp package, enabling seamless
integration with existing benchmarking and analysis tools.

The adapter handles:
- Interface compatibility between Triton and Rust backends
- Observable decoding with error detection
- Batch processing with detailed iteration tracking
- Matrix format conversion and validation
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from .decoder import RelayBPDecoder as TritonDecoder


@dataclass
class _ObsResult:
    """Result container for observable decoding with detailed statistics.
    
    Attributes:
        iterations: Number of BP iterations performed
        converged: Whether the decoder converged to a valid solution
        error_detected: Whether a logical error was detected
        observables: Predicted observable values
    """
    iterations: int
    converged: bool
    error_detected: bool
    observables: np.ndarray

class RelayDecoder:
    """Adapter for Triton Relay-BP decoder providing relay_bp interface compatibility.
    
    This class wraps the Triton GPU implementation to match the interface expected
    by the relay_bp package, enabling seamless integration with existing tools.
    
    The adapter handles matrix format conversion, parameter mapping, and provides
    the same interface as the Rust implementation for benchmarking consistency.
    """
    
    def __init__(self, check_matrix, *, error_priors, gamma0, pre_iter, num_sets,
                 set_max_iter, gamma_dist_interval=(-0.24, 0.66), stop_nconv=1,
                 stopping_criterion="nconv", logging=False, device: str = "cuda",
                 seed: Optional[int] = 0,
                 alpha: Optional[float] = None,
                 beta: Optional[float] = None,
                 dtype_messages: str = "fp32",
                 algo: Optional[str] = None,
                 perf: Optional[str] = None,
                 alpha_iteration_scaling_factor: float = 1.0,
                 bitpack_output: bool = False,
                 explicit_gammas: Optional[np.ndarray] = None,
                 **kwargs):
        # Convert input to CSR matrix format (handles various sparse matrix types)
        if sp is not None:
            self.H_csr = sp.csr_matrix(check_matrix)
        else:
            # Fallback for environments without SciPy (not recommended)
            H = np.asarray(check_matrix, dtype=np.uint8)
            from scipy.sparse import csr_matrix as _csr  # raises if SciPy missing
            self.H_csr = _csr(H)

        self.N = int(self.H_csr.shape[1])
        _plain = (num_sets == 0)
        _gamma_interval = (0.0, 0.0) if _plain else tuple(gamma_dist_interval)


        # Initialize underlying Triton decoder with mapped parameters
        self._dec = TritonDecoder(
            self.H_csr,
            error_priors=error_priors,
            pre_iter=pre_iter,
            num_sets=num_sets,
            set_max_iter=set_max_iter,
            gamma0=gamma0,
            gamma_dist_interval=_gamma_interval,
            stop_nconv=stop_nconv,
            normalized_min_sum_alpha=alpha,
            offset_min_sum_beta=beta,
            dtype_messages=dtype_messages,
            device=device,
            seed=seed,
            bitpack_output=bitpack_output,
            algo=algo,
            perf=perf,
            alpha_iteration_scaling_factor=alpha_iteration_scaling_factor,
            stopping_criterion=stopping_criterion,
            explicit_gammas=explicit_gammas,
        )


    def decode(self, syndrome: np.ndarray | torch.Tensor):
        """Decode syndrome using Triton GPU implementation.
        
        Args:
            syndrome: Syndrome vector as numpy array or torch tensor
            
        Returns:
            Dict containing decoding results with relay_bp interface compatibility
        """
        # Triton decoder expects batched input; wrap single syndrome as batch of size 1
        if isinstance(syndrome, np.ndarray):
            s = torch.from_numpy(syndrome.astype(np.uint8, copy=False)).to(self._dec.device).view(1, -1)
        else:
            s = syndrome.to(self._dec.device, dtype=torch.uint8).view(1, -1)
        
        out = self._dec.decode(s)
        bits = out["errors"][0].to(torch.int8)
        converged = bool(out["valid_mask"][0].item())
        iters = int(out.get("iterations", torch.tensor([self._dec.pre_iter], device=self._dec.device))[0].item())
        
        return {
            "bits": bits,
            "posterior_llr": None,
            "converged": converged,
            "num_ensembles": 0,
            "iterations": iters,
            "selected_idx": 0,
        }

    def decode_detailed(self, detectors: np.ndarray | torch.Tensor):
        """Decode detectors with detailed results (matches Rust interface).
        
        Args:
            detectors: Detector vector as numpy array or torch tensor
            
        Returns:
            DecodeResult object with detailed decoding information
        """
        result = self.decode(detectors)
        return _ObsResult(
            iterations=result["iterations"],
            converged=result["converged"],
            error_detected=not result["converged"],  # Error detected if not converged
            observables=result["bits"].cpu().numpy(),  # Use bits as observables for now
        )

    def decode_batch(self, detectors: np.ndarray | torch.Tensor):
        """Decode batch of detectors (matches Rust interface).
        
        Args:
            detectors: Batch of detector vectors as numpy array or torch tensor
            
        Returns:
            Batch of decoded error vectors as numpy array
        """
        if isinstance(detectors, np.ndarray):
            detectors = torch.from_numpy(detectors.astype(np.uint8, copy=False)).to(self._dec.device)
        else:
            detectors = detectors.to(self._dec.device, dtype=torch.uint8)
        
        out = self._dec.decode(detectors)
        return out["errors"].to(torch.int8).cpu().numpy()

    def decode_detailed_batch(self, detectors: np.ndarray | torch.Tensor):
        """Decode batch of detectors with detailed results (matches Rust interface).
        
        Args:
            detectors: Batch of detector vectors as numpy array or torch tensor
            
        Returns:
            List of DecodeResult objects with detailed decoding information
        """
        if isinstance(detectors, np.ndarray):
            detectors = torch.from_numpy(detectors.astype(np.uint8, copy=False)).to(self._dec.device)
        else:
            detectors = detectors.to(self._dec.device, dtype=torch.uint8)
        
        out = self._dec.decode(detectors)
        results = []
        for i in range(detectors.shape[0]):
            bits = out["errors"][i].to(torch.int8)
            converged = bool(out["valid_mask"][i].item())
            iters = int(out.get("iterations", torch.tensor([self._dec.pre_iter], device=self._dec.device))[i].item())
            
            results.append(_ObsResult(
                iterations=iters,
                converged=converged,
                error_detected=not converged,  # Error detected if not converged
                observables=bits.cpu().numpy(),  # Use bits as observables for now
            ))
        return results


class ObservableDecoderRunner:
    """Observable decoder runner providing batch processing with error detection.
    
    This class handles batch decoding of error patterns and computes observable
    predictions with logical error detection, matching the relay_bp interface.
    """
    
    def __init__(self, decoder: RelayDecoder, observables_matrix, include_decode_result: bool = True, **kwargs):
        self.decoder = decoder
        if sp is not None and sp.issparse(observables_matrix):
            self._obs = observables_matrix.tocsr()
        else:
            self._obs = sp.csr_matrix(observables_matrix) if sp is not None else np.asarray(observables_matrix, dtype=np.uint8)
        self.O, self.N = self._obs.shape
        self.M = int(decoder.H_csr.shape[0])

    def _mul_mod2(self, A_csr, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute matrix-vector multiplication modulo 2 for observable computation.
        
        Args:
            A_csr: Sparse matrix (observables or check matrix)
            x: Vector (error pattern or decoded bits)
            
        Returns:
            Result of A @ x mod 2 as uint8 array
        """
        if sp is not None and sp.issparse(A_csr):
            y = A_csr @ (np.asarray(x, dtype=np.int8))
            return (np.asarray(y) % 2).astype(np.uint8)
        A = np.asarray(A_csr, dtype=np.uint8)
        y = (A @ np.asarray(x, dtype=np.int8)) % 2
        return y.astype(np.uint8)

    def from_errors_decode_observables_detailed_batch(self, errors: np.ndarray, parallel: bool = False) -> List[_ObsResult]:
        """Decode batch of error patterns and compute observable predictions with error detection.
        
        This method implements the core benchmarking functionality, processing a batch
        of error patterns and returning detailed statistics for each pattern including
        iteration counts, convergence status, and logical error detection.
        
        Args:
            errors: [B,N] array of error patterns to decode
            parallel: Whether to use parallel processing (unused in Triton implementation)
            
        Returns:
            List of _ObsResult objects containing detailed decoding statistics
        """
        B, N = errors.shape
        assert N == self.decoder.N, f"N mismatch: errors has {N}, decoder has {self.decoder.N}"

        # Compute true observables from error patterns
        true_obs = self._mul_mod2(self._obs, errors.T).T  # (B,O)

        # Build syndromes from errors via H @ e mod 2 (vectorized)
        H = self.decoder.H_csr
        if sp is not None and sp.issparse(H):
            S = (H @ errors.T) % 2  # (M,B)
            S = np.asarray(S, dtype=np.uint8).T  # (B,M)
        else:
            H_np = np.asarray(H, dtype=np.uint8)
            S = ((H_np @ errors.T) % 2).astype(np.uint8).T

        # Batch decode on GPU
        S_gpu = torch.from_numpy(S).to(self.decoder._dec.device)
        out = self.decoder._dec.decode(S_gpu)
        bits = out["errors"].to(torch.uint8).cpu().numpy()  # (B,V)
        iters = out.get("iterations", torch.full((B,), int(self.decoder._dec.pre_iter), device=self.decoder._dec.device, dtype=torch.int32)).to(torch.int32).cpu().numpy()
        pred_obs = self._mul_mod2(self._obs, bits.T).T

        # Build results with logical error detection
        res: List[_ObsResult] = []
        valid = out["valid_mask"].to(torch.bool).cpu().numpy()
        for i in range(B):
            err_det = bool((true_obs[i] != pred_obs[i]).any())
            res.append(_ObsResult(
                iterations=int(iters[i]),
                converged=bool(valid[i]),
                error_detected=err_det,
                observables=pred_obs[i].astype(np.uint8),
            ))
        return res


