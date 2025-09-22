import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

try:
    import scipy.sparse as sp
except Exception:
    sp = None

from relay_bp_triton.decoder import RelayBPDecoder as TritonDecoder


@dataclass
class _ObsResult:
    iterations: int
    converged: bool
    error_detected: bool
    observables: np.ndarray

# relay_bp_triton_adapter.py

class RelayDecoderF64:
    def __init__(self, check_matrix, *, error_priors, gamma0, pre_iter, num_sets,
                 set_max_iter, gamma_dist_interval=(-0.24, 0.66), stop_nconv=1,
                 stopping_criterion="nconv", logging=False, device: str = "cuda",
                 seed: Optional[int] = 0):
        # Always coerce to csr_matrix (handles csr_array/coo/coo_array/etc.)
        if sp is not None:
            self.H_csr = sp.csr_matrix(check_matrix)
        else:
            # very defensive fallback; Triton path really expects SciPy present
            H = np.asarray(check_matrix, dtype=np.uint8)
            # tiny CSR shim: indices/indptr would be missing, so prefer installing SciPy
            from scipy.sparse import csr_matrix as _csr  # raises if SciPy missing
            self.H_csr = _csr(H)

        self.N = int(self.H_csr.shape[1])
        self._dec = TritonDecoder(
            self.H_csr,
            error_priors=error_priors,
            pre_iter=pre_iter,
            num_sets=num_sets,
            set_max_iter=set_max_iter,
            gamma0=gamma0,
            gamma_dist_interval=tuple(gamma_dist_interval),
            stop_nconv=stop_nconv,
            normalized_min_sum_alpha=None,
            offset_min_sum_beta=None,
            dtype_messages="fp32",
            device=device,
            seed=int(seed or 0),
            bitpack_output=False,
            mode="default",
        )


    def decode(self, syndrome: np.ndarray | torch.Tensor):
        # Triton decoder is batched; wrap 1×C
        if isinstance(syndrome, np.ndarray):
            s = torch.from_numpy(syndrome.astype(np.uint8, copy=False)).to(self._dec.device).view(1, -1)
        else:
            s = syndrome.to(self._dec.device, dtype=torch.uint8).view(1, -1)
        out = self._dec.decode(s)
        bits = out["errors"][0].to(torch.int8)
        converged = bool(out["valid_mask"][0].item())
        # Iterations proxy: use pre_iter (legs not exposed yet)
        return {
            "bits": bits,
            "posterior_llr": None,
            "converged": converged,
            "num_ensembles": 0,
            "iterations": int(self._dec.pre_iter),
            "selected_idx": 0,
        }


class ObservableDecoderRunner:
    def __init__(self, decoder: RelayDecoderF64, observables_matrix, include_decode_result: bool = True, **kwargs):
        self.decoder = decoder
        if sp is not None and sp.issparse(observables_matrix):
            self._obs = observables_matrix.tocsr()
        else:
            self._obs = sp.csr_matrix(observables_matrix) if sp is not None else np.asarray(observables_matrix, dtype=np.uint8)
        self.O, self.N = self._obs.shape
        self.M = int(decoder.H_csr.shape[0])

    def _mul_mod2(self, A_csr, x: np.ndarray | torch.Tensor) -> np.ndarray:
        # CPU path for observables/csr multiply
        if sp is not None and sp.issparse(A_csr):
            y = A_csr @ (np.asarray(x, dtype=np.int8))
            return (np.asarray(y) % 2).astype(np.uint8)
        A = np.asarray(A_csr, dtype=np.uint8)
        y = (A @ np.asarray(x, dtype=np.int8)) % 2
        return y.astype(np.uint8)

    def from_errors_decode_observables_detailed_batch(self, errors: np.ndarray, parallel: bool = False) -> List[_ObsResult]:
        B, N = errors.shape
        assert N == self.decoder.N, f"N mismatch: errors has {N}, decoder has {self.decoder.N}"

        # True observables on CPU
        true_obs = self._mul_mod2(self._obs, errors.T).T  # (B,O)

        # Build syndromes from errors via H @ e mod 2 (vectorized) on CPU
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
        pred_obs = self._mul_mod2(self._obs, bits.T).T

        res: List[_ObsResult] = []
        valid = out["valid_mask"].to(torch.bool).cpu().numpy()
        for i in range(B):
            err_det = bool((true_obs[i] != pred_obs[i]).any())
            res.append(_ObsResult(
                iterations=int(self.decoder._dec.pre_iter),  # pre_iter + legs if exposed later
                converged=bool(valid[i]),
                error_detected=err_det,
                observables=pred_obs[i].astype(np.uint8),
            ))
        return res


