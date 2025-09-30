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
from dataclasses import dataclass
import triton

from .graph import CSRGraph
from . import autotune as ac
from .kernels import (
    c2v_min_sum_kernel, v2c_and_marginals_fused_gamma_kernel,
    parity_per_check_kernel, parity_from_hard_kernel, parity_from_hard_compact_kernel, init_messages_kernel,
    be_to_eb_kernel, eb_to_be_kernel, c2v_min_sum_btile_kernel, v2c_and_gamma_btile_kernel,
    relay_decode_persistent_kernel,
    reduce_all_ge_kernel, freeze_finished_lanes_kernel, zero_check_ok_inactive_kernel,
)
from .utils import (
    compute_log_prior_ratios, sample_gamma_uniform,
    sample_gamma_scalar, validate_error_priors, validate_csr_matrix,
    bitpack_errors
)
@dataclass
class PerfCfg:
    perf: str
    msg_dtype: str
    check_every: int
    btile: int
    rows_per_chk: int
    rows_per_var: int
    c2v_block: int
    c2v_warps: int
    c2v_stages: int
    v2c_block: int
    v2c_warps: int
    v2c_stages: int

@dataclass
class RtCfg:
    Q: int
    check_every: int
    msg_dtype: str
    num_warps: int
    block_size: int



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
        bitpack_output: bool = False,
        algo: Optional[str] = None,           # "relay" or "plain"
        perf: Optional[str] = None,           # "default" | "throughput" | "realtime"
        alpha_iteration_scaling_factor: Optional[float] = 1.0,
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
        
        # Allow alpha==0.0 to enable the ramp schedule (Rust-like semantics)
        if normalized_min_sum_alpha is not None and not (0 <= normalized_min_sum_alpha <= 1):
            raise ValueError("normalized_min_sum_alpha must be in [0, 1]")
        
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
        # Normalize separate axes
        algo_norm = ((algo or ('plain' if (num_sets == 0) else 'relay'))).strip().lower()
        if algo_norm not in ("relay", "plain"):
            raise ValueError(f"Unknown algo '{algo_norm}'. Use 'relay' or 'plain'.")
        self.algo = algo_norm

        perf_norm = ((perf or 'default')).strip().lower()
        perf_aliases = {"default":"default","row":"default","rowwise":"default","throughput":"throughput","btile":"throughput","realtime":"realtime"}
        if perf_norm not in perf_aliases:
            raise ValueError(f"Unknown perf '{perf_norm}'. Use 'default'/'throughput'/'realtime'.")
        self.perf = perf_aliases[perf_norm]
        self.alpha_iteration_scaling_factor = float(alpha_iteration_scaling_factor or 1.0)
        
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
        
        # Precompute weights once (avoid recomputing each loop): match Rust decoding_quality
        # Use sum of log-priors over bits==1, ignoring infs
        wj = torch.abs(self.lambda0.squeeze(0)).to(self.device, dtype=torch.float32)  # [V]
        wj = torch.where(torch.isfinite(wj), wj, torch.zeros_like(wj))
        self.wj = wj.contiguous()
        
        # Set up message dtype (fp16 for messages, fp32 for accumulations)
        if dtype_messages == "fp16":
            self.msg_dtype = torch.float16
            self.acc_dtype = torch.float32  # Accumulations in fp32
        elif dtype_messages == "fp32":
            self.msg_dtype = torch.float32
            self.acc_dtype = torch.float32
        else:
            raise ValueError("dtype_messages must be 'fp16' or 'fp32'")
        
        # Centralize perf config
        self.cfg = PerfCfg(
            perf=self.perf,
            msg_dtype=self.dtype_messages,
            check_every=int(os.getenv("RELAY_CHECK_EVERY", "1")),
            btile=int(os.getenv("RELAY_BTILE", "16")),
            rows_per_chk=int(os.getenv("RELAY_ROWS_PER_CHK", "8")),
            rows_per_var=int(os.getenv("RELAY_ROWS_PER_VAR", "8")),
            c2v_block=0,
            c2v_warps=0,
            c2v_stages=0,
            v2c_block=0,
            v2c_warps=0,
            v2c_stages=0,
        )
        
        # Set up RNG
        torch.manual_seed(seed)
        
        # Pre-allocate device tensors (will be allocated in decode)
        self._device_tensors = {}

        # Realtime config (single env read)
        self.rt_cfg = RtCfg(
            Q=int(os.getenv("RELAY_RT_QUEUE", "64")),
            check_every=int(os.getenv("RELAY_RT_CHECK_EVERY", "8")),
            msg_dtype=os.getenv("RELAY_RT_MSG_DTYPE", "fp16"),
            num_warps=int(os.getenv("RELAY_RT_PERSIST_WARPS", "8")),
            block_size=int(os.getenv("RELAY_RT_BLOCK_SIZE", "128")),
        )
        self._rt_worker_launched = False
        # Optional logging for cached tuning usage (lightweight, guarded)
        self._tune_log = os.getenv("RELAY_TUNE_VERBOSE", "0") == "1"
        self._tune_logged: set[str] = set()
    
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
            
        }
        if self.cfg.perf == "throughput":
            # [E,B] transposed message buffers for BTILE kernels
            tensors['muT'] = torch.zeros(self.E, B, dtype=self.msg_dtype, device=self.device)
            tensors['nuT'] = torch.zeros(self.E, B, dtype=self.msg_dtype, device=self.device)
        self._device_tensors[B] = tensors
        return tensors

    def _allocate_realtime_state(self):
        if hasattr(self, "_rt_state_allocated") and self._rt_state_allocated:
            return
        Q = self.rt_cfg.Q
        # queues
        self.rt_syndromes = torch.zeros(Q, self.C, dtype=torch.uint8, device=self.device)
        self.rt_errors = torch.zeros(Q, self.V, dtype=torch.uint8, device=self.device)
        self.rt_weights = torch.zeros(Q, dtype=torch.float32, device=self.device)
        self.rt_valid = torch.zeros(Q, dtype=torch.uint8, device=self.device)
        self.rt_slot_state = torch.zeros(Q, dtype=torch.int32, device=self.device)  # 0=EMPTY,1=READY,2=DONE
        # state B=1
        msg_dtype = torch.float16 if self.rt_cfg.msg_dtype == "fp16" else torch.float32
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
            self.rt_cfg.Q, self.C, self.V, self.E,
            self.normalized_min_sum_alpha or 0.0, self.offset_min_sum_beta or 0.0,
            self.pre_iter, self.set_max_iter, self.num_sets, self.rt_cfg.check_every,
            use_alpha=(self.normalized_min_sum_alpha is not None),
            use_beta=(self.offset_min_sum_beta is not None),
            msg_is_fp16=msg_is_fp16,
            BLOCK_SIZE=self.rt_cfg.block_size,
            ROWS_PER_CHK=self.cfg.rows_per_chk,
            ROWS_PER_VAR=self.cfg.rows_per_var,
            MAX_STEPS=1000000,
            num_warps=self.rt_cfg.num_warps,
            num_stages=2,
            )
        self._rt_worker_launched = True

    def decode_rt(self, syndrome: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Realtime single-sample decode using persistent worker.
        syndrome: [1, C] uint8 on CUDA device
        """
        assert self.perf == "realtime", "Use perf='realtime' to call decode_rt()"
        if syndrome.device != torch.device(self.device):
            syndrome = syndrome.to(self.device)
        if syndrome.ndim == 1:
            syndrome = syndrome.view(1, -1)
        assert syndrome.shape == (1, self.C)
        self._ensure_worker()
        Q = self.rt_cfg.Q
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
        problem_key = {"B": B, "E": self.E, "msg_is_fp16": msg_is_fp16}
        saved = ac.try_get_saved("be_to_eb_kernel", problem_key)
        if saved is None:
            from .autotune import build_btile_transpose_configs
            cfgs = [
                dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
                for c in build_btile_transpose_configs()
            ]
            best = ac.bench_and_select(
                kernel=be_to_eb_kernel,
                grid=grid,
                args=(src_be, dst_eb, B, self.E, msg_is_fp16),
                meta_base={},
                configs=cfgs,
                number=20,
            )
            ac.set_saved("be_to_eb_kernel", problem_key, best)
            saved = best
        if self._tune_log and "be_to_eb_kernel" not in self._tune_logged:
            print(f"[relay-bp-triton] Using cached meta for be_to_eb_kernel from {ac.DEFAULT_CACHE} key={problem_key}")
            self._tune_logged.add("be_to_eb_kernel")
        kw = {}
        if 'BTILE' in saved:
            kw['BTILE'] = int(saved['BTILE'])
        be_to_eb_kernel[grid](
            src_be, dst_eb, B, self.E,
            msg_is_fp16,
            **kw,
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )

    def _eb_to_be(self, src_eb: torch.Tensor, dst_be: torch.Tensor, B: int):
        grid = (self.E,)
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        problem_key = {"B": B, "E": self.E, "msg_is_fp16": msg_is_fp16}
        saved = ac.try_get_saved("eb_to_be_kernel", problem_key)
        if saved is None:
            from .autotune import build_btile_transpose_configs
            cfgs = [
                dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
                for c in build_btile_transpose_configs()
            ]
            best = ac.bench_and_select(
                kernel=eb_to_be_kernel,
                grid=grid,
                args=(src_eb, dst_be, B, self.E, msg_is_fp16),
                meta_base={},
                configs=cfgs,
                number=20,
            )
            ac.set_saved("eb_to_be_kernel", problem_key, best)
            saved = best
        if self._tune_log and "eb_to_be_kernel" not in self._tune_logged:
            print(f"[relay-bp-triton] Using cached meta for eb_to_be_kernel from {ac.DEFAULT_CACHE} key={problem_key}")
            self._tune_logged.add("eb_to_be_kernel")
        kw = {}
        if 'BTILE' in saved:
            kw['BTILE'] = int(saved['BTILE'])
        eb_to_be_kernel[grid](
            src_eb, dst_be, B, self.E,
            msg_is_fp16,
            **kw,
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )

    def _launch_c2v_btile(self, tensors: Dict, B: int, active_idx: torch.Tensor = None, B_active: int = None):
        if active_idx is None:
            # Fallback to full batch
            active_idx = torch.arange(B, device=self.device, dtype=torch.int32)
            B_active = B
        grid = (self.C, (B_active + self.cfg.btile - 1) // self.cfg.btile)
        # Use per-iteration alpha schedule (matches _launch_c2v_kernel behavior)
        alpha = getattr(self, "_alpha_current", self.normalized_min_sum_alpha or 0.0)
        beta = self.offset_min_sum_beta or 0.0
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        problem_key = {
            "B": B, "C": self.C, "E": self.E,
            "msg_is_fp16": msg_is_fp16,
            "BTILE": self.cfg.btile,
        }
        meta_base = dict(
            B=B, B_active=B_active, C=self.C, E=self.E,
            alpha=alpha, beta=beta,
            use_alpha=use_alpha, use_beta=use_beta,
            msg_is_fp16=msg_is_fp16,
            BTILE=int(self.cfg.btile),
        )
        saved = ac.try_get_saved("c2v_min_sum_btile_kernel", problem_key)
        if saved is None:
            from .autotune import build_btile_compute_configs
            # Do NOT autotune BTILE here; keep BTILE fixed via self.cfg.btile.
            # Strip any BTILE entries from candidate configs, only tune BLOCK_SIZE/warps/stages.
            cfgs = []
            for c in build_btile_compute_configs():
                kw = dict(c.kwargs)
                if 'BTILE' in kw:
                    kw.pop('BTILE')
                cfgs.append(dict(kw, num_warps=c.num_warps, num_stages=c.num_stages))
            best = ac.bench_and_select(
                kernel=c2v_min_sum_btile_kernel,
                grid=grid,
                args=(
                    tensors['muT'], tensors['nuT'],
                    self.graph.chk_ptr, self.graph.chk_edges,
                    tensors['syndrome'], active_idx,
                ),
                meta_base=meta_base,
                configs=cfgs,
                number=20,
            )
            ac.set_saved("c2v_min_sum_btile_kernel", problem_key, best)
            saved = best
        if self._tune_log and "c2v_min_sum_btile_kernel" not in self._tune_logged:
            print(f"[relay-bp-triton] Using cached meta for c2v_min_sum_btile_kernel from {ac.DEFAULT_CACHE} key={problem_key}")
            self._tune_logged.add("c2v_min_sum_btile_kernel")
        # Fixed BTILE is already in meta_base
        kw = {'BLOCK_SIZE': int(saved['BLOCK_SIZE'])}
        c2v_min_sum_btile_kernel[grid](
            tensors['muT'], tensors['nuT'],
            self.graph.chk_ptr, self.graph.chk_edges,
            tensors['syndrome'], active_idx,
            **meta_base,
            **kw,
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )
    
    def _launch_c2v_kernel(self, tensors: Dict, B: int):
        """Launch check-to-variable kernel."""
        # alpha is set per-iteration via self._alpha_t and written to an attribute before launch
        alpha = getattr(self, "_alpha_current", self.normalized_min_sum_alpha or 0.0)
        beta = self.offset_min_sum_beta or 0.0
        use_alpha = self.normalized_min_sum_alpha is not None
        use_beta = self.offset_min_sum_beta is not None

        total = B * self.C
        grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)

        msg_is_fp16 = (self.msg_dtype == torch.float16)
        rows = self.cfg.rows_per_chk
        problem_key = {"B": B, "C": self.C, "E": self.E, "msg_is_fp16": msg_is_fp16, "ROWS_PER_CHK": rows}
        meta_base = dict(B=B, C=self.C, E=self.E, msg_is_fp16=msg_is_fp16, ROWS_PER_CHK=rows,
                         alpha=alpha, beta=beta, use_alpha=use_alpha, use_beta=use_beta)

        saved = ac.try_get_saved("c2v_min_sum_kernel", problem_key)
        if saved is None:
            from .autotune import build_c2v_configs
            cfgs = [
                dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
                for c in build_c2v_configs()
            ]
            best = ac.bench_and_select(
                kernel=c2v_min_sum_kernel,
                grid=grid,
                args=(
                    tensors['mu'], tensors['nu'],
                    self.graph.chk_ptr, self.graph.chk_edges,
                    tensors['syndrome'], tensors['active'],
                ),
                meta_base=meta_base,
                configs=cfgs,
                number=20,
            )
            ac.set_saved("c2v_min_sum_kernel", problem_key, best)
            saved = best
        if self._tune_log and "c2v_min_sum_kernel" not in self._tune_logged:
            print(f"[relay-bp-triton] Using cached meta for c2v_min_sum_kernel from {ac.DEFAULT_CACHE} key={problem_key}")
            self._tune_logged.add("c2v_min_sum_kernel")
        c2v_min_sum_kernel[grid](
            tensors['mu'], tensors['nu'],
            self.graph.chk_ptr, self.graph.chk_edges,
            tensors['syndrome'], tensors['active'],
            **meta_base,
            BLOCK_SIZE=int(saved['BLOCK_SIZE']),
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )
    
    def _launch_v2c_kernel(self, tensors: Dict, B: int):
        """Deprecated: plain V2C path removed (use fused)."""
        pass
    
    def _launch_v2c_fused_gamma_kernel(self, tensors: Dict, B: int):
        """Launch variable-to-check and marginals kernel with fused gamma mixing."""
        total = B * self.V
        grid = ((total + self.cfg.rows_per_var - 1) // self.cfg.rows_per_var,)
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        rows = self.cfg.rows_per_var
        problem_key = {"B": B, "V": self.V, "E": self.E, "msg_is_fp16": msg_is_fp16, "ROWS_PER_VAR": rows, "STORE_M": True}
        meta_base = dict(B=B, V=self.V, E=self.E, msg_is_fp16=msg_is_fp16, ROWS_PER_VAR=rows, STORE_M=True)

        saved = ac.try_get_saved("v2c_and_marginals_fused_gamma_kernel", problem_key)
        if saved is None:
            from .autotune import build_v2c_configs
            cfgs = [
                dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
                for c in build_v2c_configs()
            ]
            best = ac.bench_and_select(
                kernel=v2c_and_marginals_fused_gamma_kernel,
                grid=grid,
                args=(
                    tensors['nu'], tensors['mu'],
                    self.graph.var_ptr, self.graph.var_edges,
                    tensors['lambda_'], tensors['M'], tensors['hard_dec'],
                    tensors['active'],
                    self.lambda0, tensors['gamma'],
                ),
                meta_base=meta_base,
                configs=cfgs,
                number=20,
            )
            ac.set_saved("v2c_and_marginals_fused_gamma_kernel", problem_key, best)
            saved = best
        if self._tune_log and "v2c_and_marginals_fused_gamma_kernel" not in self._tune_logged:
            print(f"[relay-bp-triton] Using cached meta for v2c_and_marginals_fused_gamma_kernel from {ac.DEFAULT_CACHE} key={problem_key}")
            self._tune_logged.add("v2c_and_marginals_fused_gamma_kernel")
        v2c_and_marginals_fused_gamma_kernel[grid](
            tensors['nu'], tensors['mu'],
            self.graph.var_ptr, self.graph.var_edges,
            tensors['lambda_'], tensors['M'], tensors['hard_dec'],
            tensors['active'],
            self.lambda0, tensors['gamma'],
            **meta_base,
            BLOCK_SIZE=int(saved['BLOCK_SIZE']),
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )
    
    def _launch_gamma_mix_kernel(self, tensors: Dict, B: int):
        """Deprecated: gamma mixing handled in fused V2C kernel."""
        pass
    
    def _launch_parity_check_kernel(self, tensors: Dict, B: int):
        """Launch parity check kernel with simple cache-based autotuning."""
        # Sanity check: throughput mode must use parity_from_hard_kernel
        assert not (self.cfg.perf == "throughput"), "Throughput mode must use parity_from_hard_kernel"
        total = B * self.C
        grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)

        problem_key = {"B": B, "C": self.C, "V": self.V, "E": self.E, "ROWS_PER_CHK": self.cfg.rows_per_chk}
        meta_base = dict(B=B, C=self.C, V=self.V, E=self.E, ROWS_PER_CHK=self.cfg.rows_per_chk)

        saved = ac.try_get_saved("parity_per_check_kernel", problem_key)
        if saved is None:
            from .autotune import build_parity_configs
            cfgs = [
                dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
                for c in build_parity_configs()
            ]
            best = ac.bench_and_select(
                kernel=parity_per_check_kernel,
                grid=grid,
                args=(
                    tensors['M'],
                    self.graph.chk_ptr, self.graph.chk_edges, self.graph.edge_var,
                    tensors['syndrome'], tensors['check_ok'],
                ),
                meta_base=meta_base,
                configs=cfgs,
                number=20,
            )
            ac.set_saved("parity_per_check_kernel", problem_key, best)
            saved = best
        if self._tune_log and "parity_per_check_kernel" not in self._tune_logged:
            print(f"[relay-bp-triton] Using cached meta for parity_per_check_kernel from {ac.DEFAULT_CACHE} key={problem_key}")
            self._tune_logged.add("parity_per_check_kernel")
        parity_per_check_kernel[grid](
            tensors['M'],
            self.graph.chk_ptr, self.graph.chk_edges, self.graph.edge_var,
            tensors['syndrome'], tensors['check_ok'],
            **meta_base,
            BLOCK_SIZE=int(saved['BLOCK_SIZE']),
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )
    
    def _launch_init_kernel(self, tensors: Dict, B: int):
        """Launch message initialization kernel."""
        N = B * self.E
        BLOCK = 1024
        grid = ((N + BLOCK - 1) // BLOCK,)
        init_messages_kernel[grid](
            tensors['mu'], tensors['nu'],
            N,
            BLOCK=BLOCK,
        )
    
    def _launch_v2c_btile(self, tensors: Dict, B: int, write_hard: bool, store_m: bool = True, active_idx: torch.Tensor = None, B_active: int = None):
        if active_idx is None:
            # Fallback to full batch
            active_idx = torch.arange(B, device=self.device, dtype=torch.int32)
            B_active = B
        grid = (self.V, (B_active + self.cfg.btile - 1) // self.cfg.btile)
        msg_is_fp16 = (self.msg_dtype == torch.float16)
        problem_key = {"B": B, "V": self.V, "E": self.E, "msg_is_fp16": msg_is_fp16, "BTILE": self.cfg.btile, "WRITE_HARD": write_hard, "STORE_M": store_m}
        meta_base = dict(B=B, B_active=B_active, V=self.V, E=self.E, msg_is_fp16=msg_is_fp16, WRITE_HARD=write_hard, STORE_M=store_m, BTILE=int(self.cfg.btile))
        saved = ac.try_get_saved("v2c_and_gamma_btile_kernel", problem_key)
        if saved is None:
            from .autotune import build_btile_compute_configs
            # Do NOT autotune BTILE here; keep BTILE fixed via self.cfg.btile.
            # Strip any BTILE entries from candidate configs, only tune BLOCK_SIZE/warps/stages.
            cfgs = []
            for c in build_btile_compute_configs():
                kw = dict(c.kwargs)
                if 'BTILE' in kw:
                    kw.pop('BTILE')
                cfgs.append(dict(kw, num_warps=c.num_warps, num_stages=c.num_stages))
            best = ac.bench_and_select(
                kernel=v2c_and_gamma_btile_kernel,
                grid=grid,
                args=(
                    tensors['nuT'], tensors['muT'],
                    self.graph.var_ptr, self.graph.var_edges,
                    tensors['lambda_'], self.lambda0, tensors['gamma'],
                    tensors['M'], tensors['hard_dec'],
                    active_idx,
                ),
                meta_base=meta_base,
                configs=cfgs,
                number=20,
            )
            ac.set_saved("v2c_and_gamma_btile_kernel", problem_key, best)
            saved = best
        if self._tune_log and "v2c_and_gamma_btile_kernel" not in self._tune_logged:
            print(f"[relay-bp-triton] Using cached meta for v2c_and_gamma_btile_kernel from {ac.DEFAULT_CACHE} key={problem_key}")
            self._tune_logged.add("v2c_and_gamma_btile_kernel")
        # Fixed BTILE is already in meta_base
        kw = {'BLOCK_SIZE': int(saved['BLOCK_SIZE'])}
        v2c_and_gamma_btile_kernel[grid](
            tensors['nuT'], tensors['muT'],
            self.graph.var_ptr, self.graph.var_edges,
            tensors['lambda_'], self.lambda0, tensors['gamma'],
            tensors['M'], tensors['hard_dec'],
            active_idx,
            **meta_base,
            **kw,
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )
    
    def _launch_parity_from_hard_kernel(self, tensors: Dict, B: int):
        """Launch parity check using hard_dec (for throughput mode)."""
        total = B * self.C
        grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)
        problem_key = {"B": B, "C": self.C, "V": self.V, "E": self.E, "ROWS_PER_CHK": self.cfg.rows_per_chk}
        meta_base = dict(B=B, C=self.C, V=self.V, E=self.E, ROWS_PER_CHK=self.cfg.rows_per_chk)

        saved = ac.try_get_saved("parity_from_hard_kernel", problem_key)
        if saved is None:
            from .autotune import build_parity_configs
            cfgs = [dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages) for c in build_parity_configs()]
            best = ac.bench_and_select(
                kernel=parity_from_hard_kernel,
                grid=grid,
                args=(tensors['hard_dec'], self.graph.chk_ptr, self.graph.chk_edges, self.graph.edge_var,
                      tensors['syndrome'], tensors['check_ok']),
                meta_base=meta_base,
                configs=cfgs,
                number=20,
            )
            ac.set_saved("parity_from_hard_kernel", problem_key, best)
            saved = best

        parity_from_hard_kernel[grid](
            tensors['hard_dec'],
            self.graph.chk_ptr, self.graph.chk_edges, self.graph.edge_var,
            tensors['syndrome'], tensors['check_ok'],
            **meta_base,
            BLOCK_SIZE=int(saved['BLOCK_SIZE']),
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )
    
    def _launch_parity_from_hard_compact_kernel(self, tensors: Dict, B: int, active_idx: torch.Tensor, B_active: int):
        """Launch compact parity check using hard_dec for active lanes only (for throughput mode)."""
        total = B_active * self.C
        grid = ((total + self.cfg.rows_per_chk - 1) // self.cfg.rows_per_chk,)
        problem_key = {"B": B, "C": self.C, "V": self.V, "E": self.E, "ROWS_PER_CHK": self.cfg.rows_per_chk}
        meta_base = dict(B=B, B_active=B_active, C=self.C, V=self.V, E=self.E, ROWS_PER_CHK=self.cfg.rows_per_chk)

        saved = ac.try_get_saved("parity_from_hard_compact_kernel", problem_key)
        if saved is None:
            from .autotune import build_parity_configs
            cfgs = [dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages) for c in build_parity_configs()]
            best = ac.bench_and_select(
                kernel=parity_from_hard_compact_kernel,
                grid=grid,
                args=(tensors['hard_dec'], self.graph.chk_ptr, self.graph.chk_edges, self.graph.edge_var,
                      tensors['syndrome'], tensors['check_ok'], active_idx),
                meta_base=meta_base,
                configs=cfgs,
                number=20,
            )
            ac.set_saved("parity_from_hard_compact_kernel", problem_key, best)
            saved = best

        parity_from_hard_compact_kernel[grid](
            tensors['hard_dec'],
            self.graph.chk_ptr, self.graph.chk_edges, self.graph.edge_var,
            tensors['syndrome'], tensors['check_ok'], active_idx,
            **meta_base,
            BLOCK_SIZE=int(saved['BLOCK_SIZE']),
            num_warps=int(saved.get('num_warps', 0)),
            num_stages=int(saved.get('num_stages', 0)),
        )
    
    def _launch_zero_check_ok_inactive_kernel(self, tensors: Dict, B: int):
        """Zero out check_ok for inactive lanes to ensure clean parity statistics."""
        BLOCK_B = 256
        BLOCK_C = 256
        grid = ((B + BLOCK_B - 1) // BLOCK_B, (self.C + BLOCK_C - 1) // BLOCK_C)
        zero_check_ok_inactive_kernel[grid](
            tensors['check_ok'], tensors['active'], B, self.C,
            BLOCK_B=BLOCK_B, BLOCK_C=BLOCK_C,
        )
    
    def _check_parity_and_select(self, tensors: Dict, B: int, active_idx: torch.Tensor = None, B_active: int = None):
        """Check parity and select best solutions (device-only, no return)."""
        # Launch the right parity kernel
        if self.cfg.perf == "throughput":
            # parity over hard_dec (since M may not be stored)
            if active_idx is not None and B_active is not None and B_active > 0:
                # Zero out check_ok for inactive lanes first
                self._launch_zero_check_ok_inactive_kernel(tensors, B)
                # Use compact parity kernel for active lanes only
                self._launch_parity_from_hard_compact_kernel(tensors, B, active_idx, B_active)
            else:
                # Fallback to full parity kernel
                self._launch_parity_from_hard_kernel(tensors, B)
        else:
            # parity over M
            self._launch_parity_check_kernel(tensors, B)
        
        # Check if all parity constraints are satisfied
        valid = tensors['check_ok'].view(B, self.C).all(dim=1)  # [B] bool
        idx = torch.nonzero(valid, as_tuple=False).squeeze(1)  # [K]
        # Snapshot previous validity BEFORE mutating valid_solutions
        prev_valid = tensors['valid_solutions'].clone()

        if idx.numel() > 0:
            # Only compute weights for valid indices; derive current hard decisions
            if self.cfg.perf == "throughput":
                # In throughput mode, use hard_dec directly (M is not stored)
                curr_hd = tensors['hard_dec'][idx]
            else:
                # In default mode, derive from M
                curr_hd = (tensors['M'][idx] < 0).to(torch.uint8)
            cand_w = (curr_hd.float() * self.wj).sum(dim=1)  # [K]

            better = cand_w < tensors['best_weights'][idx]
            if better.any():
                upd = idx[better]
                tensors['best_weights'][upd] = cand_w[better]
                tensors['best_errors'][upd] = curr_hd[better]
                tensors['valid_solutions'][upd] = True

        # Count newly found lanes (Rust-like) using the snapshot
        newly_found = valid & ~prev_valid
        tensors['found_count'] += newly_found.to(torch.int32)
    
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
        # One-time allocations for per-lane freezing and iteration accounting
        if 'first_iter' not in tensors:
            tensors['first_iter'] = torch.full((B,), -1, dtype=torch.int32, device=self.device)
        if 'iter_counter' not in tensors:
            tensors['iter_counter'] = torch.zeros((), dtype=torch.int32, device=self.device)
        if 'active' not in tensors:
            tensors['active'] = torch.ones(B, dtype=torch.uint8, device=self.device)
        # Reset active lanes at the start of every decode
        tensors['active'].fill_(1)
        # Reset per-decode iteration accounting and frozen state
        tensors['first_iter'].fill_(-1)
        tensors['iter_counter'].zero_()
        if 'all_done_flag' not in tensors:
            tensors['all_done_flag'] = torch.zeros(1, dtype=torch.uint8, device=self.device)
        else:
            tensors['all_done_flag'].zero_()
        
        # Copy syndromes to device
        tensors['syndrome'][:] = syndromes
        
        # Initialize
        tensors['lambda_'].copy_(self.lambda0.expand(B, -1))  # Broadcast prior LLRs robustly
        tensors['best_weights'].fill_(float('inf'))
        tensors['found_count'].zero_()
        tensors['valid_solutions'].zero_()
        
        # Initialize messages
        self._launch_init_kernel(tensors, B)
        # TODO
        #tensors['mu'].zero_()
        #tensors['nu'].zero_()

        # Pre-iterations (T0) with uniform gamma0
        gamma0_tensor = sample_gamma_scalar(B, self.V, self.gamma0, self.device)
        tensors['gamma'][:] = gamma0_tensor
        
        if self.cfg.perf == "throughput":
            # transpose mu/nu to [E,B] once
            self._be_to_eb(tensors['mu'], tensors['muT'], B)
            self._be_to_eb(tensors['nu'], tensors['nuT'], B)

        for t in range(self.pre_iter):
            # Compute Rust-like alpha schedule when alpha==0.0
            if self.normalized_min_sum_alpha is None:
                self._alpha_current = 0.0
            else:
                a = float(self.normalized_min_sum_alpha)
                if a == 0.0:
                    s = float(self.alpha_iteration_scaling_factor or 1.0)
                    self._alpha_current = 1.0 - (2.0 ** (-( (t + 1) / s )))
                elif a < 0.0:
                    self._alpha_current = 1.0
                else:
                    self._alpha_current = a
            # Build active lane compaction for efficiency (reused for parity)
            active_idx = torch.nonzero(tensors['active'] != 0, as_tuple=False).squeeze(1).to(torch.int32)
            B_active = int(active_idx.numel())
            
            if self.cfg.perf == "throughput":
                # BTILE C2V and V2C fully in [E,B]
                if B_active == 0:
                    # nothing to do for BTILE kernels this iteration
                    pass
                else:
                    self._launch_c2v_btile(tensors, B, active_idx, B_active)
                    on_cadence = ((t+1) % self.cfg.check_every == 0)
                    self._launch_v2c_btile(tensors, B, write_hard=on_cadence, store_m=False, active_idx=active_idx, B_active=B_active)
            else:
                self._launch_c2v_kernel(tensors, B)
                # Fused V2C + gamma mixing updates lambda_ internally
                self._launch_v2c_fused_gamma_kernel(tensors, B)

            if (t + 1) % self.cfg.check_every == 0:
                # For parity path, we need hard_dec updated (done in v2c btile when write_hard=True)
                self._check_parity_and_select(tensors, B, active_idx, B_active)
                # Device-side freeze + first-iter capture and bump iter counter
                freeze_finished_lanes_kernel[(B,)](
                    tensors['best_errors'],
                    tensors['hard_dec'], tensors['gamma'], tensors['active'],
                    tensors['found_count'], tensors['first_iter'], tensors['iter_counter'],
                    self.stop_nconv, V=self.V,
                )
                tensors['iter_counter'] += 1
                # Compute all-done flag on device and read single byte
                reduce_all_ge_kernel[(1,)](
                    tensors['found_count'], self.stop_nconv, tensors['all_done_flag'], B
                )
                if tensors['all_done_flag'].item():
                    break
        
        # Final parity check after pre-iterations
        # In throughput mode, parity uses hard_dec snapshot; enable write and use the hard kernel
        # Build active lane compaction for efficiency (reused for parity)
        active_idx = torch.nonzero(tensors['active'] != 0, as_tuple=False).squeeze(1).to(torch.int32)
        B_active = int(active_idx.numel())
        
        if self.cfg.perf == "throughput":
            # In throughput mode, parity uses hard_dec snapshot; enable write and use the hard kernel
            if B_active > 0:
                self._launch_v2c_btile(tensors, B, write_hard=True, store_m=False, active_idx=active_idx, B_active=B_active)
        
        self._check_parity_and_select(tensors, B, active_idx, B_active)
        # Device-side freeze without bumping iter_counter
        freeze_finished_lanes_kernel[(B,)](
            tensors['best_errors'],
            tensors['hard_dec'], tensors['gamma'], tensors['active'],
            tensors['found_count'], tensors['first_iter'], tensors['iter_counter'],
            self.stop_nconv, V=self.V,
        )
        # If everyone is done, skip legs entirely and return results now
        reduce_all_ge_kernel[(1,)](
            tensors['found_count'], self.stop_nconv, tensors['all_done_flag'], B
        )
        if tensors['all_done_flag'].item():
            fallback_errors = torch.where(
                tensors['valid_solutions'].view(-1, 1),
                tensors['best_errors'],
                tensors['hard_dec'],
            )
            errors = bitpack_errors(fallback_errors) if self.bitpack_output else fallback_errors
            iters = torch.where(
                tensors['first_iter'] >= 0,
                tensors['first_iter'],
                tensors['iter_counter'].expand_as(tensors['first_iter'])
            )
            return {
                "errors": errors,
                "weights": tensors['best_weights'],
                "valid_mask": tensors['valid_solutions'],
                "iterations": iters,
            }
        
        # Relay legs (R times)
        for leg in range(self.num_sets):
            # Sample disordered gamma
            gamma_min, gamma_max = self.gamma_dist_interval
            gamma_leg = sample_gamma_uniform(B, self.V, gamma_min, gamma_max, self.device, self.seed + leg)
            tensors['gamma'][:] = gamma_leg
            
            for t in range(self.set_max_iter):
                # Per-iter alpha schedule
                if self.normalized_min_sum_alpha is None:
                    self._alpha_current = 0.0
                else:
                    a = float(self.normalized_min_sum_alpha)
                    if a == 0.0:
                        s = float(self.alpha_iteration_scaling_factor or 1.0)
                        self._alpha_current = 1.0 - (2.0 ** (-( (t + 1) / s )))
                    elif a < 0.0:
                        self._alpha_current = 1.0
                    else:
                        self._alpha_current = a
                self._launch_c2v_kernel(tensors, B)                  # produces ν
                self._launch_v2c_fused_gamma_kernel(tensors, B)
                
                # Check parity and select solutions only every CHECK_EVERY iterations
                if (t + 1) % self.cfg.check_every == 0:
                    # Build active lane compaction for efficiency (reused for parity)
                    active_idx = torch.nonzero(tensors['active'] != 0, as_tuple=False).squeeze(1).to(torch.int32)
                    B_active = int(active_idx.numel())
                    self._check_parity_and_select(tensors, B, active_idx, B_active)
                    freeze_finished_lanes_kernel[(B,)](
                        tensors['best_errors'],
                        tensors['hard_dec'], tensors['gamma'], tensors['active'],
                        tensors['found_count'], tensors['first_iter'], tensors['iter_counter'],
                        self.stop_nconv, V=self.V,
                    )
                    tensors['iter_counter'] += 1
                    reduce_all_ge_kernel[(1,)](
                        tensors['found_count'], self.stop_nconv, tensors['all_done_flag'], B
                    )
                    if tensors['all_done_flag'].item():
                        break
            
            # Final parity check after the leg
            # Build active lane compaction for efficiency (reused for parity)
            active_idx = torch.nonzero(tensors['active'] != 0, as_tuple=False).squeeze(1).to(torch.int32)
            B_active = int(active_idx.numel())
            
            if self.cfg.perf == "throughput":
                if B_active > 0:
                    self._launch_v2c_btile(tensors, B, write_hard=True, store_m=False, active_idx=active_idx, B_active=B_active)
            self._check_parity_and_select(tensors, B, active_idx, B_active)
            # Freeze without bumping iter_counter
            freeze_finished_lanes_kernel[(B,)](
                tensors['best_errors'],
                tensors['hard_dec'], tensors['gamma'], tensors['active'],
                tensors['found_count'], tensors['first_iter'], tensors['iter_counter'],
                self.stop_nconv, V=self.V,
            )
            reduce_all_ge_kernel[(1,)](
                tensors['found_count'], self.stop_nconv, tensors['all_done_flag'], B
            )
            if tensors['all_done_flag'].item():
                break
        
        # Prepare output
        # Fallback: if no valid solution was found for a batch item, use the last hard decision
        fallback_errors = torch.where(
            tensors['valid_solutions'].view(-1, 1),
            tensors['best_errors'],
            tensors['hard_dec'],
        )
        if self.bitpack_output:
            errors = bitpack_errors(fallback_errors)
        else:
            errors = fallback_errors
        
        # Build per-lane iteration counts (Rust-like): first_iter if set, else total iter_counter
        iters = tensors['first_iter'].clone()
        iters = torch.where(iters >= 0, iters, tensors['iter_counter'].expand_as(iters))

        return {
            "errors": errors,
            "weights": tensors['best_weights'],
            "valid_mask": tensors['valid_solutions'],
            "iterations": iters,
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
                "algo": self.algo,
                "perf": self.perf,
                "check_every": self.cfg.check_every,
            },
        }
