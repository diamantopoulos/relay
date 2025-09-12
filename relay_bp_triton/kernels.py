# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Triton kernels for Relay-BP-S belief propagation decoding."""

import triton
import triton.language as tl
import torch
from typing import Dict, Tuple, Optional


@triton.jit
def c2v_min_sum_kernel(
    mu, nu,                    # [B,E]
    chk_ptr, chk_edges,        # CSR over checks -> edge IDs
    syndrome,                  # [B,C]
    B, C, E,
    alpha, beta,
    use_alpha: tl.constexpr, use_beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Check-to-variable min-sum kernel with proper two-pass implementation."""
    pid = tl.program_id(axis=0)
    b = pid // C
    i = pid % C
    if (b >= B) | (i >= C):
        return

    row_start = tl.load(chk_ptr + i)
    row_end   = tl.load(chk_ptr + i + 1)
    deg = row_end - row_start
    if deg <= 0:
        return

    # ---------- PASS 1: parity + min1/min2/argmin ----------
    neg_parity = tl.zeros((), dtype=tl.int32)
    min1 = tl.full((), 1e30, dtype=tl.float32)
    min2 = tl.full((), 1e30, dtype=tl.float32)
    argmin_e = tl.full((), -1,   dtype=tl.int32)

    tile = row_start
    while tile < row_end:
        # scalar loop over a small tile
        for k in tl.static_range(0, BLOCK_SIZE):
            idx = tile + k
            m = idx < row_end
            e  = tl.load(chk_edges + idx, mask=m, other=0)
            mu_e = tl.load(mu + b*E + e,  mask=m, other=0.0)
            # sign parity
            neg_parity += tl.where(m & (mu_e < 0.0), 1, 0)
            a = tl.abs(mu_e)
            # update min1/min2/argmin
            better1 = m & (a < min1)
            # new min2 from prior min1 if we improved min1
            min2 = tl.where(better1, min1, min2)
            # or from this a if we didn't improve min1 but it's < min2
            better2 = m & (~better1) & (a < min2)
            min2 = tl.where(better2, a, min2)
            min1 = tl.where(better1, a, min1)
            argmin_e = tl.where(better1, e, argmin_e)
        tile += BLOCK_SIZE

    # degree-1: outgoing magnitude must be 0
    deg_is_one = (deg == 1)
    
    # include syndrome bit in sign product
    syn = tl.load(syndrome + b*C + i).to(tl.int32) & 1
    # parity bit = (neg_parity & 1) XOR syn
    par_bit = ((neg_parity & 1) ^ syn)
    sign_prod = tl.where(par_bit == 0, 1.0, -1.0)

    # ---------- PASS 2: write nu ----------
    tile = row_start
    while tile < row_end:
        for k in tl.static_range(0, BLOCK_SIZE):
            idx = tile + k
            m = idx < row_end
            e  = tl.load(chk_edges + idx, mask=m, other=0)
            mu_e = tl.load(mu + b*E + e, mask=m, other=0.0)
            sgn_mu = tl.where(mu_e >= 0.0, 1.0, -1.0)

            is_min1_edge = (e == argmin_e)
            out_mag = tl.where(is_min1_edge, min2, min1)
            out_mag = tl.where(deg_is_one, 0.0, out_mag)

            if use_alpha:
                out_mag = alpha * out_mag
            if use_beta:
                out_mag = tl.maximum(out_mag - beta, 0.0)

            nu_e = (sign_prod * sgn_mu) * out_mag
            tl.store(nu + b*E + e, nu_e, mask=m)
        tile += BLOCK_SIZE


@triton.jit
def v2c_and_marginals_kernel(
    nu, mu,                 # [B,E]
    var_ptr, var_edges,     # CSR over vars -> edge IDs
    lam, M, hard_dec,       # [B,V], [B,V], [B,V]
    B, V, E,
    BLOCK_SIZE: tl.constexpr,
):
    """Variable-to-check and marginals kernel with proper tiling."""
    pid = tl.program_id(axis=0)
    b = pid // V
    j = pid % V
    if (b >= B) | (j >= V):
        return

    row_start = tl.load(var_ptr + j)
    row_end   = tl.load(var_ptr + j + 1)
    lam_bj    = tl.load(lam + b*V + j)

    # PASS 1: sum incoming nu
    sum_nu = tl.zeros((), dtype=tl.float32)
    tile = row_start
    while tile < row_end:
        for k in tl.static_range(0, BLOCK_SIZE):
            idx = tile + k
            m = idx < row_end
            e  = tl.load(var_edges + idx, mask=m, other=0)
            nu_e = tl.load(nu + b*E + e, mask=m, other=0.0)
            sum_nu += tl.where(m, nu_e, 0.0)
        tile += BLOCK_SIZE

    M_bj = lam_bj + sum_nu
    tl.store(M + b*V + j, M_bj)
    tl.store(hard_dec + b*V + j, (M_bj < 0.0).to(tl.uint8))

    # PASS 2: write mu = lam + (sum_nu - nu_e)
    tile = row_start
    while tile < row_end:
        for k in tl.static_range(0, BLOCK_SIZE):
            idx = tile + k
            m = idx < row_end
            e  = tl.load(var_edges + idx, mask=m, other=0)
            nu_e = tl.load(nu + b*E + e, mask=m, other=0.0)
            mu_e = lam_bj + (sum_nu - nu_e)
            tl.store(mu + b*E + e, mu_e, mask=m)
        tile += BLOCK_SIZE


@triton.jit
def gamma_mix_kernel(
    lam0, M_prev, gamma, lam,
    B, V
):
    """Gamma mixing kernel for memory blending.
    
    Args:
        lam0: [1, V] prior LLRs (broadcast across batch)
        M_prev: [B, V] previous marginals
        gamma: [B, V] gamma values for mixing
        lam: [B, V] output LLRs
        B, V: batch size, num variables
    """
    pid = tl.program_id(axis=0)
    b = pid // V
    j = pid % V
    
    if b >= B or j >= V:
        return
    
    # Load values
    lam0_j = tl.load(lam0 + j)
    M_bj = tl.load(M_prev + b * V + j)
    g_bj = tl.load(gamma + b * V + j)
    
    # Mix: lambda = (1 - gamma) * lambda0 + gamma * M_prev
    lam_bj = (1.0 - g_bj) * lam0_j + g_bj * M_bj
    
    # Store result
    tl.store(lam + b * V + j, lam_bj)


@triton.jit
def parity_per_check_kernel(
    hard_dec,                # [B,V] uint8
    chk_ptr, chk_edges,      # CSR over checks -> edge IDs
    edge_var,                # [E] var index per edge
    syndrome,                # [B,C] uint8
    check_ok,                # [B,C] uint8 (out)
    B, C, V, E,
    BLOCK_SIZE: tl.constexpr,
):
    """Parity check kernel with proper tiling over all neighbors."""
    pid = tl.program_id(axis=0)
    b = pid // C
    i = pid % C
    if (b >= B) | (i >= C):
        return

    row_start = tl.load(chk_ptr + i)
    row_end   = tl.load(chk_ptr + i + 1)
    par = tl.zeros((), dtype=tl.int32)

    tile = row_start
    while tile < row_end:
        for k in tl.static_range(0, BLOCK_SIZE):
            idx = tile + k
            m = idx < row_end
            e  = tl.load(chk_edges + idx, mask=m, other=0)
            v  = tl.load(edge_var + e,       mask=m, other=0)
            bit = tl.load(hard_dec + b*V + v, mask=m, other=0).to(tl.int32) & 1
            par = tl.where(m, par ^ bit, par)   # XOR
        tile += BLOCK_SIZE

    syn = tl.load(syndrome + b*C + i).to(tl.int32) & 1
    ok  = (par == syn).to(tl.uint8)
    tl.store(check_ok + b*C + i, ok)


@triton.jit
def init_messages_kernel(
    mu, nu,
    B, E
):
    """Initialize messages to zero.
    
    Args:
        mu: [B, E] variable-to-check messages
        nu: [B, E] check-to-variable messages
        B, E: batch size, num edges
    """
    pid = tl.program_id(axis=0)
    b = pid // E
    e = pid % E
    
    if b >= B or e >= E:
        return
    
    tl.store(mu + b * E + e, 0.0)
    tl.store(nu + b * E + e, 0.0)


class KernelAutotuner:
    """Autotuner for kernel launch parameters."""
    
    def __init__(self):
        self.cache: Dict[Tuple, Dict] = {}
    
    def tune_c2v_kernel(self, max_deg_chk: int, dtype: torch.dtype, device: str) -> Dict:
        """Tune c2v kernel parameters."""
        key = ("c2v", max_deg_chk, str(dtype), device)
        if key in self.cache:
            return self.cache[key]
        
        # Default parameters
        config = {
            "BLOCK_SIZE": min(128, max_deg_chk),
            "num_warps": 4,
        }
        
        # TODO: Implement actual autotuning
        # For now, use heuristics based on degree
        if max_deg_chk <= 4:
            config["BLOCK_SIZE"] = 4
            config["num_warps"] = 2
        elif max_deg_chk <= 16:
            config["BLOCK_SIZE"] = 16
            config["num_warps"] = 2
        elif max_deg_chk <= 64:
            config["BLOCK_SIZE"] = 64
            config["num_warps"] = 4
        else:
            config["BLOCK_SIZE"] = 128
            config["num_warps"] = 8
        
        self.cache[key] = config
        return config
    
    def tune_v2c_kernel(self, max_deg_var: int, dtype: torch.dtype, device: str) -> Dict:
        """Tune v2c kernel parameters."""
        key = ("v2c", max_deg_var, str(dtype), device)
        if key in self.cache:
            return self.cache[key]
        
        # Default parameters
        config = {
            "BLOCK_SIZE": min(128, max_deg_var),
            "num_warps": 4,
        }
        
        # TODO: Implement actual autotuning
        if max_deg_var <= 4:
            config["BLOCK_SIZE"] = 4
            config["num_warps"] = 2
        elif max_deg_var <= 16:
            config["BLOCK_SIZE"] = 16
            config["num_warps"] = 2
        elif max_deg_var <= 64:
            config["BLOCK_SIZE"] = 64
            config["num_warps"] = 4
        else:
            config["BLOCK_SIZE"] = 128
            config["num_warps"] = 8
        
        self.cache[key] = config
        return config


# Global autotuner instance
_autotuner = KernelAutotuner()


def get_autotuner() -> KernelAutotuner:
    """Get the global autotuner instance."""
    return _autotuner
