# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Triton kernels for Relay-BP-S belief propagation decoding.

This module contains GPU kernels implementing the core belief propagation operations
for quantum error correction. The kernels closely follow the Rust implementation
in crates/relay_bp/src/bp/min_sum.rs, providing GPU acceleration for:

- Check-to-variable (C2V) message passing with min-sum algorithm
- Variable-to-check (V2C) message passing with gamma mixing
- Parity checking for syndrome validation
- Memory management and tensor operations

Key kernels:
- c2v_min_sum_kernel: Implements min-sum check node updates
- v2c_and_marginals_fused_gamma_kernel: Variable node updates with memory mixing
- parity_*_kernel: Syndrome validation kernels
- Various utility kernels for memory management and batch processing
"""

import triton
import triton.language as tl
from typing import Dict, Tuple, Optional

@triton.jit
def c2v_min_sum_kernel(
    mu, nu,                    # [B,E]
    chk_ptr, chk_edges,        # CSR over checks -> edge IDs
    syndrome,                  # [B,C]
    active,                    # [B] uint8 mask (1=active, 0=frozen)
    B, C, E,
    alpha, beta,
    use_alpha: tl.constexpr, use_beta: tl.constexpr,
    msg_is_fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Check-to-variable min-sum kernel implementing belief propagation check node updates.
    
    This kernel implements the min-sum algorithm for check nodes in belief propagation,
    corresponding to the compute_check_to_variable function in Rust min_sum.rs.
    
    For each check node, it computes outgoing messages to variable nodes using:
    1. Two-pass algorithm to find min1, min2, and parity
    2. Min-sum rule with optional alpha/beta scaling
    3. Syndrome integration for quantum error correction
    
    Args:
        mu, nu: Message tensors [B,E] (variable-to-check, check-to-variable)
        chk_ptr, chk_edges: CSR representation of check constraints
        syndrome: [B,C] syndrome bits from quantum measurements
        active: [B] mask for active decoding lanes
        B, C, E: batch size, number of checks, number of edges
        alpha, beta: min-sum scaling parameters
        use_alpha, use_beta: flags for parameter usage
        msg_is_fp16: whether messages are stored in fp16
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK
    
    # Process up to ROWS_PER_CHK {b,i} pairs
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        # Only process if within bounds
        if idx < B * C:
            b = idx // C
            i = idx % C

            # Early-out for inactive lanes - check active status first
            act_b = tl.load(active + b)
            if act_b != 0:
                # Only read CSR data for active lanes
                row_start = tl.load(chk_ptr + i)
                row_end   = tl.load(chk_ptr + i + 1)
                deg = row_end - row_start
                if deg > 0:
                    deg_is_one = deg == 1
                    # PASS 1: Compute parity and find min1/min2 values (vectorized)
                    neg_parity = tl.zeros((), dtype=tl.int32)
                    min1 = tl.full((), 1e30, dtype=tl.float32)  # fp32 for accuracy
                    min2 = tl.full((), 1e30, dtype=tl.float32)  # fp32 for accuracy
                    cnt1 = tl.zeros((), dtype=tl.int32)
                    argmin_e = tl.full((), -1,   dtype=tl.int32)

                    tile = row_start
                    while tile < row_end:
                        offs = tile + tl.arange(0, BLOCK_SIZE)
                        m    = offs < row_end
                        e    = tl.load(chk_edges + offs, mask=m, other=0)
                        mu_e = tl.load(mu + b*E + e,     mask=m, other=0.0)

                        # parity via xor of tile parity bits
                        tile_par = (tl.sum((mu_e < 0.0) & m, axis=0).to(tl.int32) & 1)
                        neg_parity = neg_parity ^ tile_par

                        a       = tl.abs(mu_e.to(tl.float32))
                        a_mask  = tl.where(m, a, 1e30)

                        # tile min1/argmin and tie count
                        pos1    = tl.argmax(-a_mask, axis=0)
                        min1_t  = tl.max(-a_mask, axis=0) * (-1.0)
                        e1_t    = tl.load(chk_edges + tile + pos1)
                        cnt1_t  = tl.sum(a_mask == min1_t, axis=0).to(tl.int32)
                        # tile min2 (mask just one occurrence of min1)
                        a_mask2 = tl.where(tl.arange(0, BLOCK_SIZE) == pos1, 1e30, a_mask)
                        min2_t  = tl.min(a_mask2, axis=0)

                        # merge into global running mins and counts
                        better1 = min1_t < min1
                        argmin_e = tl.where(better1, e1_t, argmin_e)
                        cnt1     = tl.where(better1, cnt1_t, tl.where(min1_t == min1, cnt1 + cnt1_t, cnt1))
                        min2     = tl.where(
                            better1,
                            tl.minimum(min1, min2_t),
                            tl.where(min1_t == min1, tl.minimum(min2, min2_t), tl.minimum(min2, min1_t))
                        )
                        min1     = tl.where(better1, min1_t, min1)

                        tile += BLOCK_SIZE

                    # include syndrome bit in sign product
                    syn = tl.load(syndrome + b*C + i).to(tl.int32) & 1
                    # parity bit = (neg_parity & 1) XOR syn
                    par_bit = ((neg_parity & 1) ^ syn)
                    sign_prod = tl.where(par_bit == 0, 1.0, -1.0)

                    # PASS 2: Write outgoing messages (vectorized)
                    tile = row_start
                    while tile < row_end:
                        offs = tile + tl.arange(0, BLOCK_SIZE)
                        m    = offs < row_end
                        e    = tl.load(chk_edges + offs, mask=m, other=0)
                        mu_e = tl.load(mu + b*E + e,     mask=m, other=0.0)
                        sgn_mu = tl.where(mu_e >= 0.0, 1.0, -1.0)

                        is_argmin = (e == argmin_e)
                        use_min2  = (cnt1 == 1) & is_argmin
                        out_mag   = tl.where(use_min2, min2, min1)
                        # Degree-1 checks must send zero.
                        out_mag   = tl.where(deg_is_one, 0.0, out_mag)

                        if use_alpha:
                            out_mag = alpha * out_mag
                        if use_beta:
                            out_mag = tl.maximum(out_mag - beta, 0.0)

                        nu_e = (sign_prod * sgn_mu) * out_mag
                        if msg_is_fp16:
                            tl.store(nu + b*E + e, nu_e.to(tl.float16), mask=m)
                        else:
                            tl.store(nu + b*E + e, nu_e, mask=m)

                        tile += BLOCK_SIZE
                # else: inactive lane — do nothing for this (b,i)



@triton.jit
def v2c_and_marginals_kernel(
    nu, mu, var_ptr, var_edges, lam, M, hard_dec,
    B, V, E, msg_is_fp16: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Deprecated: use v2c_and_marginals_fused_gamma_kernel instead."""
    return  # dead kernel removed; fused kernel is used instead


@triton.jit
def v2c_and_marginals_fused_gamma_kernel(
    nu, mu,                 # [B,E]
    var_ptr, var_edges,     # CSR over vars -> edge IDs
    lam, M, hard_dec,  # [B,V], [B,V], [B,V]
    active,                    # [B] uint8 mask (1=active)
    lam0, gamma,            # [1,V], [B,V] - for gamma mixing
    B, V, E,
    msg_is_fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STORE_M: tl.constexpr,  # Whether to store M (for bandwidth optimization)
    ROWS_PER_VAR: tl.constexpr,  # Number of variable rows to process per program
):
    """Variable-to-check kernel with fused gamma mixing for memory-based belief propagation.
    
    This kernel implements variable node updates in belief propagation with memory mixing,
    corresponding to the compute_variable_to_check function in Rust min_sum.rs.
    
    For each variable node, it:
    1. Computes incoming message sum from check nodes
    2. Updates marginals (beliefs) and hard decisions
    3. Applies gamma mixing: λ = (1-γ)*λ₀ + γ*M (memory-based update)
    4. Computes outgoing messages to check nodes
    
    The gamma mixing implements the "memory" aspect of Relay-BP, where previous
    beliefs are mixed with current beliefs using memory strength γ.
    
    Args:
        nu, mu: Message tensors [B,E] (check-to-variable, variable-to-check)
        var_ptr, var_edges: CSR representation of variable constraints
        lam, M, hard_dec: [B,V] beliefs, marginals, hard decisions
        active: [B] mask for active decoding lanes
        lam0, gamma: [1,V], [B,V] prior beliefs and memory strengths
        B, V, E: batch size, number of variables, number of edges
        msg_is_fp16: whether messages are stored in fp16
        BLOCK_SIZE: vectorization block size
        STORE_M: whether to store marginals (bandwidth optimization)
        ROWS_PER_VAR: number of variable rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_VAR
    
    # Process up to ROWS_PER_VAR {b,j} pairs
    for r in tl.static_range(0, ROWS_PER_VAR):
        idx = base + r
        # Only process if within bounds
        if idx < B * V:
            b = idx // V
            j = idx % V

            # Early-out for inactive lanes - check active status first
            act_b = tl.load(active + b)
            if act_b != 0:
                # Only read CSR data for active lanes
                row_start = tl.load(var_ptr + j)
                row_end   = tl.load(var_ptr + j + 1)
                lam_bj    = tl.load(lam + b*V + j)

                # PASS 1: Sum incoming messages from check nodes (vectorized)
                sum_nu = tl.zeros((), dtype=tl.float32)
                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m    = offs < row_end
                    e    = tl.load(var_edges + offs, mask=m, other=0)
                    nu_e = tl.load(nu + b*E + e,     mask=m, other=0.0)
                    sum_nu += tl.sum(tl.where(m, nu_e.to(tl.float32), 0.0), axis=0)
                    tile += BLOCK_SIZE

                M_bj_curr = lam_bj + sum_nu
                if STORE_M:
                    tl.store(M + b*V + j, M_bj_curr)  # Only store if needed
                tl.store(hard_dec + b*V + j, (M_bj_curr < 0.0).to(tl.uint8))

                # FUSED GAMMA MIXING: λ = (1-γ)*λ₀ + γ*M (memory-based belief update)
                g_bj = tl.load(gamma + b*V + j)           # [B,V]
                lam0_j = tl.load(lam0 + j)                # [1,V]
                lam_bj_new = (1.0 - g_bj) * lam0_j + g_bj * M_bj_curr
                tl.store(lam + b*V + j, lam_bj_new)

                # PASS 2: Write outgoing messages using updated beliefs (vectorized)
                tile = row_start
                while tile < row_end:
                    offs = tile + tl.arange(0, BLOCK_SIZE)
                    m    = offs < row_end
                    e    = tl.load(var_edges + offs, mask=m, other=0)
                    nu_e = tl.load(nu + b*E + e,     mask=m, other=0.0)
                    mu_e = lam_bj_new + (sum_nu - nu_e.to(tl.float32))  # Convert to fp32 for computation
                    if msg_is_fp16:
                        tl.store(mu + b*E + e, mu_e.to(tl.float16), mask=m)
                    else:
                        tl.store(mu + b*E + e, mu_e, mask=m)
                    tile += BLOCK_SIZE
            # else: inactive — do nothing (no loads, no stores)


@triton.jit
def v2c_and_marginals_mu_damped_kernel(
    nu, mu,                 # [B,E]
    var_ptr, var_edges,     # CSR over vars -> edge IDs
    lam, M, hard_dec,       # [B,V], [B,V], [B,V]
    gamma,                  # [B,V]
    B, V, E,
    msg_is_fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_VAR: tl.constexpr,
):
    """Variable node update with μ damping; λ is not modified here."""
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_VAR

    for r in tl.static_range(0, ROWS_PER_VAR):
        idx = base + r
        if idx < B * V:
            b = idx // V
            j = idx % V

            row_start = tl.load(var_ptr + j)
            row_end   = tl.load(var_ptr + j + 1)

            # sum ν
            sum_nu = tl.zeros((), dtype=tl.float32)
            tile = row_start
            while tile < row_end:
                offs = tile + tl.arange(0, BLOCK_SIZE)
                m    = offs < row_end
                e    = tl.load(var_edges + offs, mask=m, other=0)
                nu_e = tl.load(nu + b*E + e, mask=m, other=0.0)
                sum_nu += tl.sum(tl.where(m, nu_e.to(tl.float32), 0.0), axis=0)
                tile += BLOCK_SIZE

            lam_bj = tl.load(lam + b*V + j)
            M_bj = lam_bj + sum_nu
            tl.store(M + b*V + j, M_bj)
            tl.store(hard_dec + b*V + j, (M_bj < 0.0).to(tl.uint8))

            g_bj = tl.load(gamma + b*V + j)

            # write μ with damping
            tile = row_start
            while tile < row_end:
                offs = tile + tl.arange(0, BLOCK_SIZE)
                m    = offs < row_end
                e    = tl.load(var_edges + offs, mask=m, other=0)
                nu_e = tl.load(nu + b*E + e, mask=m, other=0.0)
                mu_out = lam_bj + (sum_nu - nu_e.to(tl.float32))
                mu_prev = tl.load(mu + b*E + e, mask=m, other=0.0).to(tl.float32)
                mu_new = (1.0 - g_bj) * mu_prev + g_bj * mu_out
                if msg_is_fp16:
                    tl.store(mu + b*E + e, mu_new.to(tl.float16), mask=m)
                else:
                    tl.store(mu + b*E + e, mu_new, mask=m)
                tile += BLOCK_SIZE


@triton.jit
def gamma_mix_kernel(lam0, M_prev, gamma, lam, B, V):
    """Deprecated: gamma mixing is now fused into v2c_and_marginals_fused_gamma_kernel."""
    return  # dead kernel removed; fused kernel is used instead


PARITY_CONFIGS = [
    triton.Config({'BLOCK_SIZE': 32}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
]

@triton.jit
def parity_per_check_kernel(
    M,                       # [B,V] fp32 (read-only)
    chk_ptr, chk_edges,      # CSR over checks -> edge IDs
    edge_var,                # [E] var index per edge
    syndrome,                # [B,C] uint8
    check_ok,                # [B,C] uint8 (out)
    B, C, V, E,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Parity check kernel for syndrome validation using belief marginals.
    
    This kernel validates whether the current hard decisions satisfy all check constraints,
    corresponding to the check_convergence function in Rust min_sum.rs.
    
    For each check constraint, it computes the parity of connected variable bits
    and compares with the syndrome to determine if the constraint is satisfied.
    
    Args:
        M: [B,V] belief marginals (read-only)
        chk_ptr, chk_edges: CSR representation of check constraints
        edge_var: [E] variable index for each edge
        syndrome: [B,C] syndrome bits from quantum measurements
        check_ok: [B,C] output mask for satisfied constraints
        B, C, V, E: batch size, checks, variables, edges
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK
    
    # Process up to ROWS_PER_CHK {b,i} pairs
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        # Only process if within bounds
        if idx < B * C:
            b = idx // C
            i = idx % C

            row_start = tl.load(chk_ptr + i)
            row_end   = tl.load(chk_ptr + i + 1)
            par = tl.zeros((), dtype=tl.int32)

            tile = row_start
            while tile < row_end:
                offs = tile + tl.arange(0, BLOCK_SIZE)
                m    = offs < row_end
                e    = tl.load(chk_edges + offs, mask=m, other=0)
                v    = tl.load(edge_var + e,     mask=m, other=0)
                Mv   = tl.load(M + b*V + v, mask=m, other=0.0)
                bits = (Mv < 0.0).to(tl.int32) & 1
                par  = par ^ (tl.sum(tl.where(m, bits, 0), axis=0) & 1)
                tile += BLOCK_SIZE

            syn = tl.load(syndrome + b*C + i).to(tl.int32) & 1
            ok  = (par == syn).to(tl.uint8)
            tl.store(check_ok + b*C + i, ok)


@triton.jit
def parity_from_hard_kernel(
    hard_dec,                # [B,V] uint8
    chk_ptr, chk_edges,      # CSR over checks -> edge IDs
    edge_var,                # [E] var index per edge
    syndrome,                # [B,C] uint8
    check_ok,                # [B,C] uint8 (out)
    B, C, V, E,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Parity check kernel for syndrome validation using hard decisions.
    
    This kernel validates check constraints using pre-computed hard decisions,
    optimized for throughput mode where marginals may not be stored.
    
    Args:
        hard_dec: [B,V] hard decisions (0/1 bits)
        chk_ptr, chk_edges: CSR representation of check constraints
        edge_var: [E] variable index for each edge
        syndrome: [B,C] syndrome bits from quantum measurements
        check_ok: [B,C] output mask for satisfied constraints
        B, C, V, E: batch size, checks, variables, edges
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        if idx < B * C:
            b = idx // C
            i = idx % C

            row_start = tl.load(chk_ptr + i)
            row_end   = tl.load(chk_ptr + i + 1)
            par = tl.zeros((), dtype=tl.int32)

            tile = row_start
            while tile < row_end:
                offs = tile + tl.arange(0, BLOCK_SIZE)
                m    = offs < row_end
                e    = tl.load(chk_edges + offs, mask=m, other=0)
                v    = tl.load(edge_var + e,     mask=m, other=0)
                bits = tl.load(hard_dec + b*V + v, mask=m, other=0).to(tl.int32) & 1
                par  = par ^ (tl.sum(tl.where(m, bits, 0), axis=0) & 1)
                tile += BLOCK_SIZE

            syn = tl.load(syndrome + b*C + i).to(tl.int32) & 1
            ok  = (par == syn).to(tl.uint8)
            tl.store(check_ok + b*C + i, ok)

@triton.jit
def parity_from_hard_compact_kernel(
    hard_dec,                # [B,V] uint8
    chk_ptr, chk_edges,      # CSR over checks -> edge IDs
    edge_var,                # [E] var index per edge
    syndrome,                # [B,C] uint8
    check_ok,                # [B,C] uint8 (out)
    active_idx,              # [B_active] int32 - compact list of active batch indices
    B, B_active, C, V, E,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,  # Number of check rows to process per program
):
    """Parity check kernel for active lanes only (throughput mode optimization).
    
    This kernel validates check constraints for only the active decoding lanes,
    reducing computation when many lanes have already converged.
    
    Args:
        hard_dec: [B,V] hard decisions (0/1 bits)
        chk_ptr, chk_edges: CSR representation of check constraints
        edge_var: [E] variable index for each edge
        syndrome: [B,C] syndrome bits from quantum measurements
        check_ok: [B,C] output mask for satisfied constraints
        active_idx: [B_active] indices of active batch lanes
        B, B_active, C, V, E: batch size, active batch size, checks, variables, edges
        BLOCK_SIZE: vectorization block size
        ROWS_PER_CHK: number of check rows processed per program
    """
    pid = tl.program_id(axis=0)
    base = pid * ROWS_PER_CHK
    for r in tl.static_range(0, ROWS_PER_CHK):
        idx = base + r
        if idx < B_active * C:
            b_active = idx // C
            i = idx % C
            
            # Map active batch index to actual batch index
            b = tl.load(active_idx + b_active)

            row_start = tl.load(chk_ptr + i)
            row_end   = tl.load(chk_ptr + i + 1)
            par = tl.zeros((), dtype=tl.int32)

            tile = row_start
            while tile < row_end:
                offs = tile + tl.arange(0, BLOCK_SIZE)
                m    = offs < row_end
                e    = tl.load(chk_edges + offs, mask=m, other=0)
                v    = tl.load(edge_var + e,     mask=m, other=0)
                bits = tl.load(hard_dec + b*V + v, mask=m, other=0).to(tl.int32) & 1
                par  = par ^ (tl.sum(tl.where(m, bits, 0), axis=0) & 1)
                tile += BLOCK_SIZE

            syn = tl.load(syndrome + b*C + i).to(tl.int32) & 1
            ok  = (par == syn).to(tl.uint8)
            tl.store(check_ok + b*C + i, ok)

@triton.jit
def zero_check_ok_inactive_kernel(
    check_ok,    # [B,C] uint8
    active,      # [B] uint8
    B, C,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Zero out check_ok for inactive lanes to ensure clean parity statistics.
    
    This utility kernel ensures that inactive decoding lanes have zero check_ok values,
    preventing them from contributing to convergence statistics.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    c_idx = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mb = b_idx < B
    mc = c_idx < C

    # load active flags for this tile's rows
    act = tl.load(active + b_idx, mask=mb, other=0)
    inactive_rows = (act == 0) & mb

    # broadcast to 2D tile
    bb = b_idx[:, None]
    cc = c_idx[None, :]
    mask2d = inactive_rows[:, None] & mc[None, :]

    tl.store(check_ok + bb * C + cc, 0, mask=mask2d)


@triton.jit
def check_and_select_kernel(
    check_ok,              # [B,C] uint8
    M,                     # [B,V] fp32 (may be unused if USE_M==0)
    hard_dec,              # [B,V] uint8
    wj,                    # [V] fp32
    best_weights,          # [B] fp32 (in/out)
    best_errors,           # [B,V] uint8 (out when improved)
    valid_solutions,       # [B] uint8 (out flag)
    found_count,           # [B] int32 (in/out)
    B, C, V,
    USE_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Device-side solution selection and validation kernel.
    
    This kernel performs solution selection and validation entirely on device,
    avoiding host-device transfers. For each lane, it:
    1. Checks if all parity constraints are satisfied
    2. Computes solution weight using log-prior ratios
    3. Updates best solution if current solution is better
    
    This corresponds to the solution selection logic in Rust relay.rs.
    
    Args:
        check_ok: [B,C] mask of satisfied constraints
        M: [B,V] belief marginals (may be unused if USE_M==0)
        hard_dec: [B,V] hard decisions
        wj: [V] log-prior ratios for weight computation
        best_weights: [B] current best weights (in/out)
        best_errors: [B,V] current best error patterns (out)
        valid_solutions: [B] valid solution flags (out)
        found_count: [B] convergence counters (in/out)
        B, C, V: batch size, checks, variables
        USE_M: whether to use marginals or hard decisions for weight computation
        BLOCK_C, BLOCK_V: vectorization block sizes
    """
    b = tl.program_id(axis=0)
    if b >= B:
        return

    # 1) all_ok = all(check_ok[b, :])
    all_ok = tl.full((), 1, dtype=tl.int32)
    c0 = 0
    while c0 < C:
        cc = c0 + tl.arange(0, BLOCK_C)
        mc = cc < C
        vals = tl.load(check_ok + b * C + cc, mask=mc, other=0).to(tl.int32)
        # if any zero present in this tile, min will be 0
        tile_all = tl.min(tl.where(mc, vals, 1), axis=0)
        all_ok = tl.minimum(all_ok, tile_all)
        c0 += BLOCK_C

    if all_ok == 0:
        return

    # 2) compute current hard decisions and weight
    weight = tl.zeros((), dtype=tl.float32)
    v0 = 0
    while v0 < V:
        vv = v0 + tl.arange(0, BLOCK_V)
        mv = vv < V
        if USE_M:
            Mv = tl.load(M + b * V + vv, mask=mv, other=0.0)
            bits = (Mv < 0.0).to(tl.uint8)
        else:
            bits = tl.load(hard_dec + b * V + vv, mask=mv, other=0)
        wtile = tl.load(wj + vv, mask=mv, other=0.0)
        weight += tl.sum(wtile * bits.to(tl.float32), axis=0)
        v0 += BLOCK_V

    # 3) compare-and-update best
    prev_best = tl.load(best_weights + b)
    better = weight < prev_best
    if better:
        tl.store(best_weights + b, weight)
        prev_valid = tl.load(valid_solutions + b)
        # write best_errors row
        off = 0
        while off < V:
            vv = off + tl.arange(0, BLOCK_V)
            mv = vv < V
            if USE_M:
                Mv2 = tl.load(M + b * V + vv, mask=mv, other=0.0)
                bits2 = (Mv2 < 0.0).to(tl.uint8)
            else:
                bits2 = tl.load(hard_dec + b * V + vv, mask=mv, other=0)
            tl.store(best_errors + b * V + vv, bits2, mask=mv)
            off += BLOCK_V
        # mark valid
        tl.store(valid_solutions + b, 1)
        if prev_valid == 0:
            old = tl.load(found_count + b)
            tl.store(found_count + b, old + 1)

@triton.jit
def init_messages_kernel(
    mu, nu,
    N,
    BLOCK: tl.constexpr,
):
    """Initialize message tensors to zero.
    
    This utility kernel performs vectorized zero initialization of message tensors,
    corresponding to the initialization in Rust min_sum.rs.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    z = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(mu + offs, z, mask=m)
    tl.store(nu + offs, z, mask=m)


@triton.jit
def reduce_all_ge_kernel(
    found_count,      # [B] int32
    stop_nconv,       # scalar int32
    out_flag,         # [1] uint8 (out)
    B                 # scalar int32
):
    """Check if all lanes have converged (device-side reduction).
    
    This kernel performs a device-side reduction to check if all decoding lanes
    have found enough valid solutions to stop early.
    """
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    acc = tl.full((), 1, dtype=tl.int32)
    tile = 0
    while tile < B:
        offs = tile + tl.arange(0, 256)
        m    = offs < B
        fc   = tl.load(found_count + offs, mask=m, other=stop_nconv)
        ok   = (fc >= stop_nconv).to(tl.int32)
        acc  = tl.minimum(acc, tl.min(tl.where(m, ok, 1), axis=0))
        tile += 256
    tl.store(out_flag, acc.to(tl.uint8))


@triton.jit
def freeze_finished_lanes_kernel(
    best_errors,           # [B,V] uint8
    hard_dec, gamma, active,                # [B,V] uint8, [B,V] fp32, [B] uint8
    found_count, first_iter, iter_counter,  # [B] int32, [B] int32, scalar int32
    stop_nconv, V: tl.constexpr
):
    """Freeze converged lanes and capture convergence iteration.
    
    This kernel implements lane freezing for converged solutions, corresponding
    to the convergence handling in Rust relay.rs. For lanes that have found
    enough valid solutions:
    1. Records the first convergence iteration
    2. Freezes the lane (sets active=0)
    3. Resets gamma values to prevent further updates
    
    Args:
        best_errors: [B,V] best error patterns found so far
        hard_dec: [B,V] current hard decisions (updated for converged lanes)
        gamma: [B,V] memory strengths (reset to 0 for converged lanes)
        active: [B] active lane mask (set to 0 for converged lanes)
        found_count: [B] convergence counters
        first_iter: [B] first convergence iteration (updated for converged lanes)
        iter_counter: scalar current iteration counter
        stop_nconv: target number of solutions for convergence
        V: number of variables
    """
    b = tl.program_id(axis=0)
    fc = tl.load(found_count + b)
    done_now = (fc >= stop_nconv)
    if done_now:
        fi = tl.load(first_iter + b)
        if fi < 0:
            itc = tl.load(iter_counter)
            tl.store(first_iter + b, itc)
            col = 0
            while col < V:
                offs = col + tl.arange(0, 128)
                m = offs < V
                be = tl.load(best_errors + b * V + offs, mask=m, other=0)
                tl.store(hard_dec + b * V + offs, be, mask=m)
                col += 128
        tl.store(active + b, 0)
        col2 = 0
        while col2 < V:
            offs2 = col2 + tl.arange(0, 128)
            m2 = offs2 < V
            tl.store(gamma + b * V + offs2, 0.0, mask=m2)
            col2 += 128

@triton.jit
def be_to_eb_kernel(
    src_be, dst_eb,
    B, E,
    msg_is_fp16: tl.constexpr,
    BTILE: tl.constexpr,
):
    """Transpose messages from [B,E] to [E,B] in BTILE chunks of B."""
    e = tl.program_id(axis=0)
    if e >= E:
        return
    b0 = 0
    while b0 < B:
        bs = b0 + tl.arange(0, BTILE)
        m = bs < B
        if msg_is_fp16:
            vals = tl.load(src_be + bs * E + e, mask=m, other=0).to(tl.float16)
            tl.store(dst_eb + e * B + bs, vals, mask=m)
        else:
            vals = tl.load(src_be + bs * E + e, mask=m, other=0.0)
            tl.store(dst_eb + e * B + bs, vals, mask=m)
        b0 += BTILE


@triton.jit
def eb_to_be_kernel(
    src_eb, dst_be,
    B, E,
    msg_is_fp16: tl.constexpr,
    BTILE: tl.constexpr,
):
    """Transpose messages from [E,B] to [B,E] in BTILE chunks of B."""
    e = tl.program_id(axis=0)
    if e >= E:
        return
    b0 = 0
    while b0 < B:
        bs = b0 + tl.arange(0, BTILE)
        m = bs < B
        vals = tl.load(src_eb + e * B + bs, mask=m, other=0)
        if msg_is_fp16:
            tl.store(dst_be + bs * E + e, vals.to(tl.float16), mask=m)
        else:
            tl.store(dst_be + bs * E + e, vals, mask=m)
        b0 += BTILE


@triton.jit
def c2v_min_sum_btile_kernel(
    muT, nuT,                 # [E,B]
    chk_ptr, chk_edges,       # CSR over checks -> edge IDs
    syndrome,                 # [B,C] (uint8)
    active_idx,               # [B_active] int32 - compact list of active batch indices
    B, B_active, C, E,
    alpha, beta,
    use_alpha: tl.constexpr, use_beta: tl.constexpr,
    msg_is_fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BTILE: tl.constexpr,
):
    """Check-to-variable min-sum over [E,B] layout, vectorized across BTILE batch lanes.
    Each program processes one check row i and updates all its edges' nuT[:, b:b+BTILE]."""
    i = tl.program_id(axis=0)
    btile_id = tl.program_id(axis=1)
    if i >= C:
        return
    b0 = btile_id * BTILE
    pos = b0 + tl.arange(0, BTILE)
    mpos = pos < B_active
    bs = tl.load(active_idx + pos, mask=mpos, other=0)
    mb = mpos  # replace old mb; no need to AND with active[] anymore

    row_start = tl.load(chk_ptr + i)
    row_end   = tl.load(chk_ptr + i + 1)
    deg = row_end - row_start
    if deg <= 0:
        return
    deg_is_one = (deg == 1)

    # Accumulators per batch lane
    neg_parity = tl.zeros((BTILE,), dtype=tl.int32)
    min1 = tl.full((BTILE,), 1e30, dtype=tl.float32)
    min2 = tl.full((BTILE,), 1e30, dtype=tl.float32)
    cnt1 = tl.zeros((BTILE,), dtype=tl.int32)

    # PASS 1: compute min1/min2 and parity per batch lane
    tile = row_start
    while tile < row_end:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        mrow = offs < row_end
        evec = tl.load(chk_edges + offs, mask=mrow, other=0)             # [BLOCK_SIZE]
        # Build 2D mask and gather mu over edges×BTILE
        mask2d = (mrow[:, None]) & (mb[None, :])
        mu_tile = tl.load(muT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)  # [BLOCK_SIZE, BTILE]
        # tile parity via xor of tile parity bits
        tile_par = (tl.sum((mu_tile < 0.0) & mask2d, axis=0).to(tl.int32) & 1)
        neg_parity = neg_parity ^ tile_par
        a = tl.abs(mu_tile.to(tl.float32))
        a = tl.where(mask2d, a, 1e30)
        # min1 over edges and tie count
        min1_t = tl.min(a, axis=0)
        cnt1_t = tl.sum(a == min1_t[None, :], axis=0).to(tl.int32)
        # min2: mask a single occurrence along edge axis
        pos1   = tl.argmax(-a, axis=0)
        a2     = tl.where(tl.arange(0, BLOCK_SIZE)[:, None] == pos1[None, :], 1e30, a)
        min2_t = tl.min(a2, axis=0)
        # merge with running min1/min2 and counts
        better1 = min1_t < min1
        cnt1 = tl.where(better1, cnt1_t, tl.where(min1_t == min1, cnt1 + cnt1_t, cnt1))
        min2 = tl.where(better1, tl.minimum(min1, min2_t), tl.where(min1_t == min1, tl.minimum(min2, min2_t), tl.minimum(min2, min1_t)))
        min1 = tl.where(better1, min1_t, min1)
        tile += BLOCK_SIZE

    syn = tl.load(syndrome + bs * C + i, mask=mb, other=0).to(tl.int32) & 1
    par_bit = ((neg_parity & 1) ^ syn)
    sign_prod = tl.where(par_bit == 0, 1.0, -1.0)

    # PASS 2: compute and store nu over edges×BTILE
    tile = row_start
    while tile < row_end:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        mrow = offs < row_end
        evec = tl.load(chk_edges + offs, mask=mrow, other=0)
        mask2d = (mrow[:, None]) & (mb[None, :])
        mu_tile = tl.load(muT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
        sgn = tl.where(mu_tile >= 0.0, 1.0, -1.0)
        # Identify min1 lanes per edge×batch and apply correct tie logic
        a = tl.abs(mu_tile.to(tl.float32))
        a_mask = tl.where(mask2d, a, 1e30)
        is_min1 = (a_mask == min1[None, :])
        use_min2 = (cnt1[None, :] == 1) & is_min1
        out_mag = tl.where(use_min2, min2[None, :], min1[None, :])
        # Degree-1 checks must send zero.
        out_mag = tl.where(deg_is_one, 0.0, out_mag)
        if use_alpha:
            out_mag = alpha * out_mag
        if use_beta:
            out_mag = tl.maximum(out_mag - beta, 0.0)
        nu_tile = (sign_prod[None, :] * sgn) * out_mag
        if msg_is_fp16:
            tl.store(nuT + (evec[:, None]) * B + bs[None, :], nu_tile.to(tl.float16), mask=mask2d)
        else:
            tl.store(nuT + (evec[:, None]) * B + bs[None, :], nu_tile, mask=mask2d)
        tile += BLOCK_SIZE


@triton.jit
def v2c_and_gamma_btile_kernel(
    nuT, muT,                 # [E,B]
    var_ptr, var_edges,       # CSR over vars -> edge IDs
    lam, lam0, gamma,         # [B,V], [V], [B,V]
    M, hard_dec,              # [B,V], [B,V]
    active_idx,               # [B_active] int32 - compact list of active batch indices
    B, B_active, V, E,
    msg_is_fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BTILE: tl.constexpr,
    WRITE_HARD: tl.constexpr,
    STORE_M: tl.constexpr,
):
    """Variable-to-check + gamma mixing on [E,B] layout, vectorized across BTILE.
    For each variable j and batch tile bs, compute sum_nu, M, lambda update, and write muT.
    """
    j = tl.program_id(axis=0)
    btile_id = tl.program_id(axis=1)
    if j >= V:
        return
    b0 = btile_id * BTILE
    pos = b0 + tl.arange(0, BTILE)
    mpos = pos < B_active
    bs = tl.load(active_idx + pos, mask=mpos, other=0)
    mb = mpos  # replace old mb; no need to AND with active[] anymore

    row_start = tl.load(var_ptr + j)
    row_end   = tl.load(var_ptr + j + 1)

    # sum incoming nu over edges for batch lanes (vectorized)
    sum_nu = tl.zeros((BTILE,), dtype=tl.float32)
    tile = row_start
    while tile < row_end:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        mrow = offs < row_end
        evec = tl.load(var_edges + offs, mask=mrow, other=0)                 # [BLOCK_SIZE]
        mask2d = (mrow[:, None]) & (mb[None, :])
        nu_tile = tl.load(nuT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
        sum_nu += tl.sum(tl.where(mask2d, nu_tile.to(tl.float32), 0.0), axis=0)
        tile += BLOCK_SIZE

    # beliefs and gamma mix (fp32 compute)
    lam_bj = tl.load(lam + bs * V + j, mask=mb, other=0.0)
    M_bj = lam_bj + sum_nu
    # Conditionally store M to avoid bandwidth in hot path
    if STORE_M:
        tl.store(M + bs * V + j, M_bj, mask=mb)
    # Optionally update hard_dec if requested (to avoid bandwidth in hot path)
    if WRITE_HARD:
        tl.store(hard_dec + bs * V + j, (M_bj < 0.0).to(tl.uint8), mask=mb)
    g_bj = tl.load(gamma + bs * V + j, mask=mb, other=0.0)
    lam0_j = tl.load(lam0 + j)
    lam_new = (1.0 - g_bj) * lam0_j + g_bj * M_bj
    tl.store(lam + bs * V + j, lam_new, mask=mb)

    # write muT for each edge lane (vectorized)
    tile = row_start
    while tile < row_end:
        offs = tile + tl.arange(0, BLOCK_SIZE)
        mrow = offs < row_end
        evec = tl.load(var_edges + offs, mask=mrow, other=0)
        mask2d = (mrow[:, None]) & (mb[None, :])
        nu_tile = tl.load(nuT + (evec[:, None]) * B + bs[None, :], mask=mask2d, other=0.0)
        mu_tile = lam_new[None, :] + (sum_nu[None, :] - nu_tile.to(tl.float32))
        if msg_is_fp16:
            tl.store(muT + (evec[:, None]) * B + bs[None, :], mu_tile.to(tl.float16), mask=mask2d)
        else:
            tl.store(muT + (evec[:, None]) * B + bs[None, :], mu_tile, mask=mask2d)
        tile += BLOCK_SIZE


@triton.jit
def relay_decode_persistent_kernel(
    chk_ptr, chk_edges, var_ptr, var_edges, edge_var, edge_chk,
    mu, nu, lam, M, gamma, hard_dec, check_ok,
    lambda0,
    rt_syndromes, rt_errors, rt_weights, rt_valid, rt_slot_state,
    Q, C, V, E,
    alpha, beta,
    PRE_ITERS, SET_ITERS, NUM_SETS, CHECK_EVERY,
    use_alpha: tl.constexpr, use_beta: tl.constexpr,
    msg_is_fp16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CHK: tl.constexpr,
    ROWS_PER_VAR: tl.constexpr,
    MAX_STEPS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    slot = pid
    for _ in tl.static_range(0, MAX_STEPS):
        slot = slot + 1
        slot = tl.where(slot >= Q, 0, slot)
        state = tl.load(rt_slot_state + slot)
        work = state == 1
        if work:
            syn_base = slot * C
            v0 = tl.arange(0, BLOCK_SIZE)
            off_v = 0
            while off_v < V:
                vv = off_v + v0
                mv = vv < V
                tl.store(lam + vv, tl.load(lambda0 + vv, mask=mv, other=0.0), mask=mv)
                tl.store(hard_dec + vv, 0, mask=mv)
                off_v += BLOCK_SIZE
            e0 = tl.arange(0, BLOCK_SIZE)
            off_e = 0
            while off_e < E:
                ee = off_e + e0
                me = ee < E
                if msg_is_fp16:
                    z16 = tl.zeros((BLOCK_SIZE,), dtype=tl.float16)
                    tl.store(mu + ee, z16, mask=me)
                    tl.store(nu + ee, z16, mask=me)
                else:
                    z32 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                    tl.store(mu + ee, z32, mask=me)
                    tl.store(nu + ee, z32, mask=me)
                off_e += BLOCK_SIZE

            stop_now = 0
            t = 0
            while t < PRE_ITERS:
                # C2V
                ci = 0
                while ci < C:
                    r = 0
                    while (r < ROWS_PER_CHK) & ((ci + r) < C):
                        i = ci + r
                        row_start = tl.load(chk_ptr + i)
                        row_end   = tl.load(chk_ptr + i + 1)
                        deg = row_end - row_start
                        if deg > 0:
                            neg_par = tl.zeros((), dtype=tl.int32)
                            min1 = tl.full((), 1e30, dtype=tl.float32)
                            min2 = tl.full((), 1e30, dtype=tl.float32)
                            tile = row_start
                            while tile < row_end:
                                offs = tile + tl.arange(0, BLOCK_SIZE)
                                mrow = offs < row_end
                                evec = tl.load(chk_edges + offs, mask=mrow, other=0)
                                mu_tile = tl.load(mu + evec, mask=mrow, other=0.0)
                                neg_par += tl.sum((mu_tile < 0.0) & mrow, axis=0).to(tl.int32)
                                a = tl.abs(mu_tile.to(tl.float32))
                                a_mask = tl.where(mrow, a, 1e30)
                                min1_t = tl.min(a_mask, axis=0)
                                a2 = tl.where(a_mask == min1_t, 1e30, a_mask)
                                min2_t = tl.min(a2, axis=0)
                                better1 = min1_t < min1
                                min2 = tl.where(better1, tl.minimum(min1, min2_t), tl.minimum(min2, min1_t))
                                min1 = tl.where(better1, min1_t, min1)
                                tile += BLOCK_SIZE
                            syn_i = tl.load(rt_syndromes + syn_base + i).to(tl.int32) & 1
                            par_bit = ((neg_par & 1) ^ syn_i)
                            sign_prod = tl.where(par_bit == 0, 1.0, -1.0)
                            tile2 = row_start
                            while tile2 < row_end:
                                offs = tile2 + tl.arange(0, BLOCK_SIZE)
                                mrow = offs < row_end
                                evec = tl.load(chk_edges + offs, mask=mrow, other=0)
                                mu_tile = tl.load(mu + evec, mask=mrow, other=0.0)
                                sgn = tl.where(mu_tile >= 0.0, 1.0, -1.0)
                                out_mag = tl.where(deg == 1, 0.0, min1)
                                if use_alpha:
                                    out_mag = alpha * out_mag
                                if use_beta:
                                    out_mag = tl.maximum(out_mag - beta, 0.0)
                                nu_tile = (sign_prod * sgn) * out_mag
                                if msg_is_fp16:
                                    tl.store(nu + evec, nu_tile.to(tl.float16), mask=mrow)
                                else:
                                    tl.store(nu + evec, nu_tile, mask=mrow)
                                tile2 += BLOCK_SIZE
                        r += 1
                    ci += ROWS_PER_CHK
                # V2C + gamma
                vj = 0
                while vj < V:
                    r = 0
                    while (r < ROWS_PER_VAR) & ((vj + r) < V):
                        j = vj + r
                        row_start = tl.load(var_ptr + j)
                        row_end   = tl.load(var_ptr + j + 1)
                        sum_nu = tl.zeros((), dtype=tl.float32)
                        tile = row_start
                        while tile < row_end:
                            offs = tile + tl.arange(0, BLOCK_SIZE)
                            mrow = offs < row_end
                            evec = tl.load(var_edges + offs, mask=mrow, other=0)
                            nu_tile = tl.load(nu + evec, mask=mrow, other=0.0)
                            sum_nu += tl.sum(tl.where(mrow, nu_tile.to(tl.float32), 0.0), axis=0)
                            tile += BLOCK_SIZE
                        lam_j = tl.load(lam + j)
                        M_j = lam_j + sum_nu
                        tl.store(M + j, M_j)
                        tl.store(hard_dec + j, (M_j < 0.0).to(tl.uint8))
                        g_j = tl.load(gamma + j)
                        lam0_j = tl.load(lambda0 + j)
                        lam_new = (1.0 - g_j) * lam0_j + g_j * M_j
                        tl.store(lam + j, lam_new)
                        tile2 = row_start
                        while tile2 < row_end:
                            offs = tile2 + tl.arange(0, BLOCK_SIZE)
                            mrow = offs < row_end
                            evec = tl.load(var_edges + offs, mask=mrow, other=0)
                            nu_tile = tl.load(nu + evec, mask=mrow, other=0.0)
                            mu_tile = lam_new + (sum_nu - nu_tile.to(tl.float32))
                            if msg_is_fp16:
                                tl.store(mu + evec, mu_tile.to(tl.float16), mask=mrow)
                            else:
                                tl.store(mu + evec, mu_tile, mask=mrow)
                            tile2 += BLOCK_SIZE
                        r += 1
                    vj += ROWS_PER_VAR
                # parity cadence
                if ((t + 1) % CHECK_EVERY) == 0:
                    all_ok = 1
                    ci2 = 0
                    while ci2 < C:
                        i = ci2
                        row_start = tl.load(chk_ptr + i)
                        row_end   = tl.load(chk_ptr + i + 1)
                        par = tl.zeros((), dtype=tl.int32)
                        tilep = row_start
                        while tilep < row_end:
                            offs = tilep + tl.arange(0, BLOCK_SIZE)
                            mrow = offs < row_end
                            evec = tl.load(chk_edges + offs, mask=mrow, other=0)
                            vvec = tl.load(edge_var + evec, mask=mrow, other=0)
                            bits = tl.load(hard_dec + vvec, mask=mrow, other=0).to(tl.int32) & 1
                            par = par ^ (tl.sum(tl.where(mrow, bits, 0), axis=0) & 1)
                            tilep += BLOCK_SIZE
                        syn_i = tl.load(rt_syndromes + syn_base + i).to(tl.int32) & 1
                        ok = (par == syn_i).to(tl.uint8)
                        tl.store(check_ok + i, ok)
                        all_ok = all_ok & ok.to(tl.int32)
                        ci2 += 1
                    stop_now = tl.where(all_ok == 1, 1, stop_now)
                t += 1
            # relay legs
            leg = 0
            while leg < NUM_SETS:
                if stop_now == 0:
                    tt = 0
                    while tt < SET_ITERS:
                        if stop_now == 0:
                            # c2v
                            ci = 0
                            while ci < C:
                                r = 0
                                while (r < ROWS_PER_CHK) & ((ci + r) < C):
                                    i = ci + r
                                    row_start = tl.load(chk_ptr + i)
                                    row_end   = tl.load(chk_ptr + i + 1)
                                    deg = row_end - row_start
                                    if deg > 0:
                                        neg_par = tl.zeros((), dtype=tl.int32)
                                        min1 = tl.full((), 1e30, dtype=tl.float32)
                                        min2 = tl.full((), 1e30, dtype=tl.float32)
                                        tile = row_start
                                        while tile < row_end:
                                            offs = tile + tl.arange(0, BLOCK_SIZE)
                                            mrow = offs < row_end
                                            evec = tl.load(chk_edges + offs, mask=mrow, other=0)
                                            mu_tile = tl.load(mu + evec, mask=mrow, other=0.0)
                                            neg_par += tl.sum((mu_tile < 0.0) & mrow, axis=0).to(tl.int32)
                                            a = tl.abs(mu_tile.to(tl.float32))
                                            a_mask = tl.where(mrow, a, 1e30)
                                            min1_t = tl.min(a_mask, axis=0)
                                            a2 = tl.where(a_mask == min1_t, 1e30, a_mask)
                                            min2_t = tl.min(a2, axis=0)
                                            better1 = min1_t < min1
                                            min2 = tl.where(better1, tl.minimum(min1, min2_t), tl.minimum(min2, min1_t))
                                            min1 = tl.where(better1, min1_t, min1)
                                            tile += BLOCK_SIZE
                                        syn_i = tl.load(rt_syndromes + syn_base + i).to(tl.int32) & 1
                                        par_bit = ((neg_par & 1) ^ syn_i)
                                        sign_prod = tl.where(par_bit == 0, 1.0, -1.0)
                                        tile2 = row_start
                                        while tile2 < row_end:
                                            offs = tile2 + tl.arange(0, BLOCK_SIZE)
                                            mrow = offs < row_end
                                            evec = tl.load(chk_edges + offs, mask=mrow, other=0)
                                            mu_tile = tl.load(mu + evec, mask=mrow, other=0.0)
                                            sgn = tl.where(mu_tile >= 0.0, 1.0, -1.0)
                                            out_mag = tl.where(deg == 1, 0.0, min1)
                                            if use_alpha:
                                                out_mag = alpha * out_mag
                                            if use_beta:
                                                out_mag = tl.maximum(out_mag - beta, 0.0)
                                            nu_tile = (sign_prod * sgn) * out_mag
                                            if msg_is_fp16:
                                                tl.store(nu + evec, nu_tile.to(tl.float16), mask=mrow)
                                            else:
                                                tl.store(nu + evec, nu_tile, mask=mrow)
                                            tile2 += BLOCK_SIZE
                                    r += 1
                                ci += ROWS_PER_CHK
                            # v2c+gamma
                            vj = 0
                            while vj < V:
                                r = 0
                                while (r < ROWS_PER_VAR) & ((vj + r) < V):
                                    j = vj + r
                                    row_start = tl.load(var_ptr + j)
                                    row_end   = tl.load(var_ptr + j + 1)
                                    sum_nu = tl.zeros((), dtype=tl.float32)
                                    tile = row_start
                                    while tile < row_end:
                                        offs = tile + tl.arange(0, BLOCK_SIZE)
                                        mrow = offs < row_end
                                        evec = tl.load(var_edges + offs, mask=mrow, other=0)
                                        nu_tile = tl.load(nu + evec, mask=mrow, other=0.0)
                                        sum_nu += tl.sum(tl.where(mrow, nu_tile.to(tl.float32), 0.0), axis=0)
                                        tile += BLOCK_SIZE
                                    lam_j = tl.load(lam + j)
                                    M_j = lam_j + sum_nu
                                    tl.store(M + j, M_j)
                                    tl.store(hard_dec + j, (M_j < 0.0).to(tl.uint8))
                                    g_j = tl.load(gamma + j)
                                    lam0_j = tl.load(lambda0 + j)
                                    lam_new = (1.0 - g_j) * lam0_j + g_j * M_j
                                    tl.store(lam + j, lam_new)
                                    tile2 = row_start
                                    while tile2 < row_end:
                                        offs = tile2 + tl.arange(0, BLOCK_SIZE)
                                        mrow = offs < row_end
                                        evec = tl.load(var_edges + offs, mask=mrow, other=0)
                                        nu_tile = tl.load(nu + evec, mask=mrow, other=0.0)
                                        mu_tile = lam_new + (sum_nu - nu_tile.to(tl.float32))
                                        if msg_is_fp16:
                                            tl.store(mu + evec, mu_tile.to(tl.float16), mask=mrow)
                                        else:
                                            tl.store(mu + evec, mu_tile, mask=mrow)
                                        tile2 += BLOCK_SIZE
                                    r += 1
                                vj += ROWS_PER_VAR
                            if ((tt + 1) % CHECK_EVERY) == 0:
                                all_ok = 1
                                ci2 = 0
                                while ci2 < C:
                                    i = ci2
                                    row_start = tl.load(chk_ptr + i)
                                    row_end   = tl.load(chk_ptr + i + 1)
                                    par = tl.zeros((), dtype=tl.int32)
                                    tilep = row_start
                                    while tilep < row_end:
                                        offs = tilep + tl.arange(0, BLOCK_SIZE)
                                        mrow = offs < row_end
                                        evec = tl.load(chk_edges + offs, mask=mrow, other=0)
                                        vvec = tl.load(edge_var + evec, mask=mrow, other=0)
                                        bits = tl.load(hard_dec + vvec, mask=mrow, other=0).to(tl.int32) & 1
                                        par = par ^ (tl.sum(tl.where(mrow, bits, 0), axis=0) & 1)
                                        tilep += BLOCK_SIZE
                                    syn_i = tl.load(rt_syndromes + syn_base + i).to(tl.int32) & 1
                                    ok = (par == syn_i).to(tl.uint8)
                                    tl.store(check_ok + i, ok)
                                    all_ok = all_ok & ok.to(tl.int32)
                                    ci2 += 1
                                stop_now = tl.where(all_ok == 1, 1, stop_now)
                        tt += 1
                # final parity after leg
                if stop_now == 0:
                    all_ok = 1
                    ci2 = 0
                    while ci2 < C:
                        i = ci2
                        row_start = tl.load(chk_ptr + i)
                        row_end   = tl.load(chk_ptr + i + 1)
                        par = tl.zeros((), dtype=tl.int32)
                        tilep = row_start
                        while tilep < row_end:
                            offs = tilep + tl.arange(0, BLOCK_SIZE)
                            mrow = offs < row_end
                            evec = tl.load(chk_edges + offs, mask=mrow, other=0)
                            vvec = tl.load(edge_var + evec, mask=mrow, other=0)
                            bits = tl.load(hard_dec + vvec, mask=mrow, other=0).to(tl.int32) & 1
                            par = par ^ (tl.sum(tl.where(mrow, bits, 0), axis=0) & 1)
                            tilep += BLOCK_SIZE
                        syn_i = tl.load(rt_syndromes + syn_base + i).to(tl.int32) & 1
                        ok = (par == syn_i).to(tl.uint8)
                        tl.store(check_ok + i, ok)
                        all_ok = all_ok & ok.to(tl.int32)
                        ci2 += 1
                    stop_now = tl.where(all_ok == 1, 1, stop_now)
                leg += 1
            # write outputs
            off = 0
            base_v = slot * V
            while off < V:
                vv = off + tl.arange(0, BLOCK_SIZE)
                mv = vv < V
                vals = tl.load(hard_dec + vv, mask=mv, other=0)
                tl.store(rt_errors + base_v + vv, vals, mask=mv)
                off += BLOCK_SIZE
            tl.store(rt_weights + slot, 0.0)
            tl.store(rt_valid + slot, (stop_now == 1).to(tl.uint8))
            tl.store(rt_slot_state + slot, 2)

