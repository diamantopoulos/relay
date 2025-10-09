#!/usr/bin/env python3
"""
Relay-BP Detailed Decoder Script

This script provides detailed analysis of Relay-BP and Plain-BP decoding performance,
measuring iteration counts and per-cycle error rates to match the paper's methodology.
It supports both Rust and Triton backends with comprehensive benchmarking capabilities.

The script implements the exact methodology from the Relay-BP paper:
- X-axis: Average BP iteration count (not wall time)
- Y-axis: Per-cycle logical error rate (not per-shot)
- Parameters: S = solutions sought, R = max relay legs

Key features:
- Detailed iteration counting for both Relay-BP and Plain-BP
- Deterministic reproducibility with configurable seeding
- Support for multiple backends (Rust/Triton) and data types
- Comprehensive timing analysis (decoder-only and end-to-end)
- Multiple output formats (simple, JSON, CSV)
"""

import argparse
import time
import numpy as np
import sys
import random
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent / "tests"))
from testdata import get_test_circuit, filter_detectors_by_basis

import stim
import relay_bp
from relay_bp.stim.sinter.check_matrices import CheckMatrices
import hashlib
import json
from relay_bp_triton.utils import describe_check_matrices

def _select_backend(backend: str, dtype: str = 'fp32', require: str | None = None):
    """Select appropriate decoder classes for the chosen backend and data type.
    
    This function provides a unified interface for selecting decoder implementations
    across different backends (Rust/Triton) and data types (fp16/fp32/fp64).
    
    Args:
        backend: "rust" or "triton" 
        dtype: "fp16", "fp32", or "fp64" (availability depends on backend)
        require: "relay", "plain", or None (which decoder types to require)
        
    Returns:
        Tuple of (RelayDecoder, ObservableDecoderRunner, MinSumBPDecoder)
        MinSumBPDecoder is None for Triton backend
        
    Raises:
        ValueError: If requested dtype/backend combination is unavailable
    """
    if backend == "triton":
        if dtype not in ("fp16", "fp32"):
            raise ValueError("Triton supports only dtype in {'fp16','fp32'}")
        from relay_bp_triton.adapter import RelayDecoder as _RelayDecoderAdapter
        from relay_bp_triton.adapter import ObservableDecoderRunner as _ObservableDecoderRunner
        return _RelayDecoderAdapter, _ObservableDecoderRunner, None
    # rust (dtype-selected, mode-specific requirement)
    import relay_bp as _rb
    if dtype not in ("fp32", "fp64"):
        raise ValueError("Rust backend supports only dtype in {'fp32','fp64'} (no fp16 exports present)")
    relay_cls = None
    minsum_cls = None
    # dtype suffix mapping to Rust class names
    suffix_map = {"fp16": "F16", "fp32": "F32", "fp64": "F64"}
    suff = suffix_map[dtype]
    if require in (None, 'relay'):
        relay_name = f"RelayDecoder{suff}"
        if not hasattr(_rb, relay_name):
            raise ValueError(f"Requested Rust relay decoder '{relay_name}' not available. Install a build with this dtype.")
        relay_cls = getattr(_rb, relay_name)
    if require in (None, 'plain'):
        minsum_name = f"MinSumBPDecoder{suff}"
        if not hasattr(_rb, minsum_name):
            raise ValueError(f"Requested Rust plain decoder '{minsum_name}' not available. Install a build with this dtype.")
        minsum_cls = getattr(_rb, minsum_name)
    from relay_bp import ObservableDecoderRunner as _ObservableDecoderRunner
    return relay_cls, _ObservableDecoderRunner, minsum_cls


def _format_time_units(ns: float) -> str:
    """Format time in nanoseconds to a compact multi-unit string.
    
    Args:
        ns: Time in nanoseconds
        
    Returns:
        Formatted string showing time in ns, µs, ms, and seconds
    """
    us = ns / 1e3
    ms = ns / 1e6
    s = ns / 1e9
    return f"{ns:.1f} ns | {us:.3f} µs | {ms:.3f} ms | {s:.6f} s"


def _hash_update_arr(hasher: "hashlib._blake2.blake2b", arr) -> None:
    """Update BLAKE2b hasher with array data for deterministic reproducibility.
    
    This function ensures deterministic results by hashing all relevant data
    in a consistent order and format, enabling reproducible experiments.
    
    Args:
        hasher: BLAKE2b hasher instance
        arr: NumPy array to hash
    """
    a = np.ascontiguousarray(arr)
    hasher.update(a.tobytes(order="C"))


def parse_args():
    """Parse command line arguments for Relay-BP detailed analysis.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Run Relay-BP decoder with detailed iteration counting')
    
    # Mode parameter
    parser.add_argument('--mode', choices=['relay','plain'], default='relay')
    parser.add_argument('--perf', choices=['default','throughput','realtime'], default='default',
                       help='Kernel execution path (default=row-wise)')
    parser.add_argument('--dtype', choices=['fp16','fp32','fp64'], default='fp32',
                       help='Numeric dtype: applies to Triton (fp16/fp32) and Rust (fp16/fp32/fp64 where available)')

    # Circuit parameters
    parser.add_argument('--circuit', type=str, default='bicycle_bivariate_144_12_12_memory_choi_XZ',
                       help='Circuit name (default: bicycle_bivariate_144_12_12_memory_Z)')
    parser.add_argument('--basis', type=str, choices=['z', 'xz'], default='xz', help='Decoding basis (default: xz)')
    parser.add_argument('--distance', type=int, default=12, help='Code distance (default: 12)')
    parser.add_argument('--rounds', type=int, default=12, help='Number of rounds (default: 12)')
    parser.add_argument('--error-rate', type=float, default=0.003, help='Physical error rate (default: 0.003)')
    parser.add_argument('--target-errors', type=int, default=10, help='Target number of errors to collect (default: 10)')
    parser.add_argument('--batch', type=int, default=2000, help='Batch size for error collection (default: 2000)')
    parser.add_argument('--max-shots', type=int, default=1000000, help='Maximum shots to collect (default: 1000000)')
    
    # Relay-BP parameters
    parser.add_argument('--gamma0', type=float, default=0.125, help='Ordered memory parameter (default: 0.125)')
    parser.add_argument('--pre-iter', type=int, default=80, help='Pre-iterations (default: 80)')
    parser.add_argument('--num-sets', type=int, default=3, help='Number of Relay-BP sets (default: 10)')
    parser.add_argument('--set-max-iter', type=int, default=60, help='Max iterations per set (default: 60)')
    parser.add_argument('--gamma-dist-min', type=float, default=-0.24, help='Gamma distribution minimum (default: -0.24)')
    parser.add_argument('--gamma-dist-max', type=float, default=0.66, help='Gamma distribution maximum (default: 0.66)')
    parser.add_argument('--stop-nconv', type=int, default=1, help='Stop after N converged solutions (default: 1)')
    parser.add_argument('--alpha', type=float, default=None, help='(plain) Normalized min-sum alpha; None for standard')    
    
    # Execution parameters
    parser.add_argument('--parallel', action='store_true', help='Enable Relay-BP builtin parallelism (default: False)')
    parser.add_argument('--output-format', type=str, choices=['simple', 'json', 'csv'], default='simple',
                       help='Output format (default: simple)')
    parser.add_argument('--seed', type=int, default=0, help='Global seed for all random number generators (default: 0)')
    parser.add_argument('--backend', type=str, choices=['rust', 'triton'], default=None,
                       help='Decoder backend to use: rust (default) or triton')
    # Always measure and report execution time unconditionally
    
    return parser.parse_args()


def run_relay_bp_detailed(args):
    """Run Relay-BP experiment with detailed iteration counting.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dict containing experiment results and statistics
    """
    return run_relay_bp_experiment(
        circuit=args.circuit,
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds,
        error_rate=args.error_rate,
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        gamma_dist_min=args.gamma_dist_min,
        gamma_dist_max=args.gamma_dist_max,
        stop_nconv=args.stop_nconv,
        target_errors=args.target_errors,
        batch=args.batch,
        max_shots=args.max_shots,
        parallel=args.parallel,
        seed=args.seed,
        backend=args.backend,
        perf=args.perf,
        dtype=args.dtype,
    )


def run_relay_bp_experiment(circuit, basis, distance, rounds, error_rate, gamma0, pre_iter, 
                           num_sets, set_max_iter, gamma_dist_min, gamma_dist_max, stop_nconv,
                           target_errors=20, batch=2000, max_shots=1000000, parallel=True, seed=0,
                           backend: str | None = None, perf: str = "default", dtype: str = 'fp32'):
    """Run Relay-BP experiment with detailed iteration counting and performance analysis.
    
    This function implements the core Relay-BP benchmarking methodology, measuring
    iteration counts and logical error rates to match the paper's analysis.
    
    Args:
        circuit: Circuit name/identifier
        basis: Decoding basis ('z' or 'xz')
        distance: Code distance
        rounds: Number of syndrome measurement rounds
        error_rate: Physical error rate
        gamma0: Ordered memory parameter (γ₀)
        pre_iter: Pre-iterations (T₀)
        num_sets: Number of relay sets (R)
        set_max_iter: Max iterations per set (Tr)
        gamma_dist_min/max: Disordered memory parameter range
        stop_nconv: Stop after N converged solutions (S)
        target_errors: Target number of logical errors to collect
        batch: Batch size for error sampling
        max_shots: Maximum shots to collect
        parallel: Enable parallel processing
        seed: Random seed for reproducibility
        backend: Decoder backend ('rust' or 'triton')
        perf: Performance mode ('default', 'throughput', 'realtime')
        dtype: Data type ('fp16', 'fp32', 'fp64')
        
    Returns:
        Dict containing comprehensive experiment results including:
        - Logical error rates (per-shot and per-cycle)
        - Average BP iterations (paper's x-axis)
        - Timing metrics (decoder-only and end-to-end)
        - Determinism hash for reproducibility verification
    """
    
    # Deterministic seeding using global seed
    random.seed(seed)
    np.random.seed(seed)

    # Get test circuit
    circuit_obj = get_test_circuit(
        circuit=circuit, 
        distance=distance, 
        rounds=rounds, 
        error_rate=error_rate
    )
    
    # Apply basis filtering based on argument
    if basis == 'z':
        circuit_obj = filter_detectors_by_basis(circuit_obj, "Z")
    # For XZ decoding, don't filter (keep all observables)
    
    # Create detector error model
    dem = circuit_obj.detector_error_model()
    
    # Print detector error model statistics
    print("DEM:",
        "detectors M =", dem.num_detectors,
        "observables O =", dem.num_observables,
        "errors N ≈", CheckMatrices.from_dem(dem).check_matrix.shape[1])
    print("Per-cycle detectors ≈", dem.num_detectors // rounds)
    print("Basis =", basis)

    print(f"Using rounds={rounds} as the number of cycles for per-cycle LER conversion")
    # Validate circuit configuration
    if isinstance(circuit, str):
        tag_ok = f"_{rounds}_" in circuit
        if not tag_ok:
            print(f"[warn] Circuit name '{circuit}' does not obviously encode rounds={rounds}.")
    assert dem.num_observables > 0, (
        "No observables after basis filtering. "
        "Use XZ (no filter) to match the paper."
    )
    
    # Create check matrices and initialize reproducibility hasher
    check_matrices = CheckMatrices.from_dem(dem)
    hasher = hashlib.blake2b(digest_size=32)
    
    # Hash problem specification for deterministic reproducibility
    cm = check_matrices.check_matrix
    om = check_matrices.observables_matrix
    _hash_update_arr(hasher, cm.indptr.astype(np.int64, copy=False))
    _hash_update_arr(hasher, cm.indices.astype(np.int64, copy=False))
    _hash_update_arr(hasher, om.indptr.astype(np.int64, copy=False))
    _hash_update_arr(hasher, om.indices.astype(np.int64, copy=False))
    _hash_update_arr(hasher, check_matrices.error_priors.astype(np.float64, copy=False))
    
    # Hash configuration and seeds
    cfg_txt = json.dumps({
        "circuit": circuit, "basis": basis, "distance": distance, "rounds": rounds,
        "error_rate": error_rate, "gamma0": gamma0, "pre_iter": pre_iter,
        "num_sets": num_sets, "set_max_iter": set_max_iter,
        "gamma_dist": [gamma_dist_min, gamma_dist_max],
        "stop_nconv": stop_nconv, "parallel": bool(parallel),
        "stim_seed": seed, "relay_seed": seed,
    }, sort_keys=True).encode()
    hasher.update(cfg_txt)
    
    # Create Relay-BP decoder with backend-specific configuration
    _RelayDecoderDyn, _ObservableDecoderRunner, _ = _select_backend(backend, dtype, require='relay')
    relay_common_kwargs = dict(
        error_priors=check_matrices.error_priors,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        gamma_dist_interval=(gamma_dist_min, gamma_dist_max),
        stop_nconv=stop_nconv,
        stopping_criterion="nconv",
        logging=False,
    )
    
    # Backend-specific decoder initialization
    if backend == 'triton':
        decoder = _RelayDecoderDyn(
            check_matrices.check_matrix,
            **relay_common_kwargs,
            algo=("relay" if num_sets > 0 else "plain"),
            perf=perf,
            dtype_messages=('fp16' if dtype == 'fp16' else 'fp32'),
        )
    else:
        decoder = _RelayDecoderDyn(
            check_matrices.check_matrix,
            **relay_common_kwargs,
        )
    observable_decoder = _ObservableDecoderRunner(
        decoder,
        check_matrices.observables_matrix,
        include_decode_result=True,
    )
    
    # Initialize statistics tracking
    avg_bp_iterations = 0.0
    avg_legs = 0.0
    legs_all = []
    total_decode_ns = 0
    total_decode_shots = 0
    total_shots = 0
    total_errors = 0
    bp_iters_all = []
    
    # Create error sampler and start timing
    sampler = dem.compile_sampler(seed=seed)
    print(f"Collecting until {target_errors} errors or {max_shots} shots...")
    start_time = time.time()
    
    while total_errors < target_errors and total_shots < max_shots:
        # Sample errors and decode with detailed iteration tracking
        det, obs, errors = sampler.sample(batch, return_errors=True)
        
        # Decode with timing measurement
        t0_ns = time.perf_counter_ns()
        obs_det = observable_decoder.from_errors_decode_observables_detailed_batch(
            errors.astype(np.uint8), parallel=parallel
        )
        t1_ns = time.perf_counter_ns()
        if t0_ns is not None and t1_ns is not None:
            total_decode_ns += (t1_ns - t0_ns)
            total_decode_shots += len(obs_det)
        
        # Update reproducibility hash
        _hash_update_arr(hasher, errors.astype(np.uint8, copy=False))
        O = check_matrices.observables_matrix.shape[0]
        pred_obs = np.empty((len(obs_det), O), dtype=np.uint8)
        for i, r in enumerate(obs_det):
            pred_obs[i, :] = np.asarray(r.observables, dtype=np.uint8)
        _hash_update_arr(hasher, pred_obs)
        
        # Extract iteration statistics
        iters_arr = np.array([r.iterations for r in obs_det], dtype=float)
        iters_i32 = np.fromiter((r.iterations for r in obs_det), dtype=np.int32, count=len(obs_det))
        succ_arr  = np.fromiter((r.converged for r in obs_det), dtype=np.uint8, count=len(obs_det))
        fail_arr  = np.fromiter((r.error_detected for r in obs_det), dtype=np.uint8, count=len(obs_det))
        _hash_update_arr(hasher, iters_i32)
        _hash_update_arr(hasher, succ_arr)
        _hash_update_arr(hasher, fail_arr)
        
        # Track BP iterations (paper's x-axis) and relay legs
        bp_iters_all.extend(iters_arr.tolist())
        legs_all.extend(1.0 + np.maximum(0.0, (iters_arr - pre_iter) / set_max_iter))
        
        # Count logical errors and provide diagnostic breakdown
        errs = int(sum(r.error_detected for r in obs_det))
        n = len(obs_det)
        n_fail = errs
        n_conv = int(sum(r.converged for r in obs_det))
        n_ok = int(sum((not r.error_detected) and r.converged for r in obs_det))
        n_bad_but_conv = int(sum(r.error_detected and r.converged for r in obs_det))
        n_bad_and_nconv = n_fail - n_bad_but_conv
        n_ok_and_nconv = int(sum((not r.error_detected) and (not r.converged) for r in obs_det))
        print(f"    batch breakdown: success&conv={n_ok}, fail&conv={n_bad_but_conv}, fail&nconv={n_bad_and_nconv}, success&nconv={n_ok_and_nconv}, conv={n_conv}/{n}")
        total_errors += errs
        total_shots += batch
        
        print(f"  Collected {total_shots} shots, {total_errors} errors (batch errors: {errs})")
        
        # Debug output for first batch
        if total_shots <= 2000:
            print(f"    Sample error_detected: {[r.error_detected for r in obs_det[:3]]}")
            print(f"    Sample iterations: {[r.iterations for r in obs_det[:3]]}")
    
    logical_error_rate = total_errors / total_shots if total_shots > 0 else 0.0
    
    # Calculate per-cycle logical error rate (as per paper)
    per_cycle_logical_error_rate = 1 - (1 - logical_error_rate) ** (1 / rounds)

    obs_count = dem.num_observables
    # For CSS-style XZ decoding, obs_count ≈ 2*k; for single-basis Z or X, obs_count ≈ k.
    if basis.lower() == 'xz':
        k_guess = obs_count // 2
    else:
        k_guess = obs_count

    per_round_per_qubit_rate = logical_error_rate / (rounds * max(1, k_guess))

    # Optional alt check: if rounds had been half-cycles, this would be different
    per_cycle_ler_half_rounds = 1 - (1 - logical_error_rate) ** (1 / max(1, 2 * rounds))
    
    # Calculate averages from collected data
    if bp_iters_all:
        avg_bp_iterations = float(np.mean(bp_iters_all))
        if legs_all:
            avg_legs = float(np.mean(legs_all))
    else:
        avg_bp_iterations = 0.0
        avg_legs = 0.0
    
    # Calculate runtime per shot (end-to-end) and decoder-only metrics
    runtime_per_shot = None
    decoder_runtime_per_shot = None
    decoder_runtime_per_iteration = None
    wall_time_s = None
    decoder_runtime_per_leg = None
    end_time = time.time()
    if start_time and end_time and total_shots > 0:
        total_runtime_seconds = end_time - start_time
        runtime_per_shot = (total_runtime_seconds * 1e9) / total_shots  # Convert to nanoseconds
    if total_decode_shots > 0:
        decoder_runtime_per_shot = total_decode_ns / total_decode_shots
        total_iterations = float(np.sum(bp_iters_all)) if bp_iters_all else 0.0
        if total_iterations > 0:
            decoder_runtime_per_iteration = total_decode_ns / total_iterations
        if legs_all:
            total_legs = float(np.sum(legs_all))
            if total_legs > 0:
                decoder_runtime_per_leg = total_decode_ns / total_legs
    
    # Prepare results (convert numpy types to Python types for JSON serialization)
    results_data = {
        'shots': int(total_shots),
        'logical_errors': int(total_errors),
        'logical_error_rate': float(logical_error_rate),
        'per_cycle_logical_error_rate': float(per_cycle_logical_error_rate),
        'per_round_per_qubit_rate': float(per_round_per_qubit_rate),
        'avg_bp_iterations': float(avg_bp_iterations),  # Paper's x-axis
        'avg_legs': float(avg_legs),  # For debugging/reporting only
        'config': {
            'gamma0': float(gamma0),
            'num_sets': int(num_sets),
            'stop_nconv': int(stop_nconv),
            'gamma_dist_interval': (float(gamma_dist_min), float(gamma_dist_max)),
            'circuit': str(circuit),
            'distance': int(distance),
            'rounds': int(rounds),
            'error_rate': float(error_rate),
            'batch': int(batch),
        }
    }
    
    if runtime_per_shot is not None:
        results_data['runtime_per_shot_ns'] = float(runtime_per_shot)
    if decoder_runtime_per_shot is not None:
        results_data['decoder_runtime_per_shot_ns'] = float(decoder_runtime_per_shot)
    if decoder_runtime_per_iteration is not None:
        results_data['decoder_runtime_per_iteration_ns'] = float(decoder_runtime_per_iteration)
    if decoder_runtime_per_leg is not None:
        results_data['decoder_runtime_per_leg_ns'] = float(decoder_runtime_per_leg)
    # Attach optional diagnostic to help verify cycle root choice
    results_data['per_cycle_ler_alt_half_rounds'] = float(per_cycle_ler_half_rounds)
    # Finalize determinism hash
    results_data['determinism_hash'] = hasher.hexdigest()
    
    return results_data


def run_plain_bp_detailed(args):
    """Run Plain-BP experiment with detailed iteration counting.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dict containing experiment results and statistics
    """
    return run_plain_bp_experiment(
        circuit=args.circuit,
        basis=args.basis,
        distance=args.distance,
        rounds=args.rounds,
        error_rate=args.error_rate,
        set_max_iter=args.set_max_iter,
        alpha=args.alpha,
        target_errors=args.target_errors,
        batch=args.batch,
        max_shots=args.max_shots,
        parallel=args.parallel,
        seed=args.seed,
        backend=args.backend,
        perf=args.perf,
        dtype=args.dtype,
    )

def run_plain_bp_experiment(circuit, basis, distance, rounds, error_rate,
                            set_max_iter=200, alpha=None, target_errors=20, batch=2000,
                            max_shots=1000000, parallel=True, seed=0,
                            backend: str | None = None, perf: str = "default", algo: str = "plain",
                            dtype: str = 'fp16'):
    """Run Plain-BP experiment with detailed iteration counting and performance analysis.
    
    This function implements Plain-BP (standard min-sum belief propagation without
    memory or relay mechanisms) for comparison with Relay-BP performance.
    
    Args:
        circuit: Circuit name/identifier
        basis: Decoding basis ('z' or 'xz')
        distance: Code distance
        rounds: Number of syndrome measurement rounds
        error_rate: Physical error rate
        set_max_iter: Maximum BP iterations
        alpha: Normalized min-sum parameter (None for standard min-sum)
        target_errors: Target number of logical errors to collect
        batch: Batch size for error sampling
        max_shots: Maximum shots to collect
        parallel: Enable parallel processing
        seed: Random seed for reproducibility
        backend: Decoder backend ('rust' or 'triton')
        perf: Performance mode ('default', 'throughput', 'realtime')
        algo: Algorithm type ('plain')
        dtype: Data type ('fp16', 'fp32', 'fp64')
        
    Returns:
        Dict containing comprehensive experiment results including:
        - Logical error rates (per-shot and per-cycle)
        - Average BP iterations
        - Timing metrics (decoder-only and end-to-end)
        - Determinism hash for reproducibility verification
    """

    # Initialize deterministic seeding
    random.seed(seed)
    np.random.seed(seed)
    try:
        # Stim seeding - try different methods depending on version
        if hasattr(stim, 'set_global_seed'):
            stim.set_global_seed(seed)
        elif hasattr(stim, 'set_seed'):
            stim.set_seed(seed)
    except Exception:
        pass

    circuit_obj = get_test_circuit(
        circuit=circuit,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
    )

    if basis == 'z':
        circuit_obj = filter_detectors_by_basis(circuit_obj, "Z")

    dem = circuit_obj.detector_error_model()
    print(f"DEM stats: detectors={dem.num_detectors}, observables={dem.num_observables}")
    print(f"Using rounds={rounds} as the number of cycles for per-cycle LER conversion")
    assert dem.num_observables > 0

    check_matrices = CheckMatrices.from_dem(dem)
    stats = describe_check_matrices(
        check_matrices,
        dem=dem,
        rounds=rounds,
        compute_components=True,  # set True if you want components (may be slow for huge H)
        compute_rank=True,        # set True if you can afford GF(2) rank
        return_dict=True,
    )


    # Initialize determinism hasher (32-byte BLAKE2b)
    hasher = hashlib.blake2b(digest_size=32)
    # Problem spec: H, observables, priors
    cm = check_matrices.check_matrix
    om = check_matrices.observables_matrix
    _hash_update_arr(hasher, cm.indptr.astype(np.int64, copy=False))
    _hash_update_arr(hasher, cm.indices.astype(np.int64, copy=False))
    _hash_update_arr(hasher, om.indptr.astype(np.int64, copy=False))
    _hash_update_arr(hasher, om.indices.astype(np.int64, copy=False))
    _hash_update_arr(hasher, check_matrices.error_priors.astype(np.float64, copy=False))
    # Config + seeds
    cfg_txt = json.dumps({
        "circuit": circuit, "basis": basis, "distance": distance, "rounds": rounds,
        "error_rate": error_rate, "set_max_iter": set_max_iter, "alpha": alpha,
        "parallel": bool(parallel), "stim_seed": seed
    }, sort_keys=True).encode()
    hasher.update(cfg_txt)

    # Create Plain-BP decoder with backend-specific configuration
    if backend == "triton":
        # Plain BP via Triton adapter (degenerate Relay-BP with no memory)
        from relay_bp_triton.adapter import RelayDecoder as _RelayDecoderFXX
        from relay_bp_triton.adapter import ObservableDecoderRunner as _ObservableDecoderRunner
        decoder = _RelayDecoderFXX(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            gamma0=0.0,
            pre_iter=set_max_iter,
            num_sets=0,
            set_max_iter=0,
            gamma_dist_interval=(0.0, 0.0),
            stop_nconv=1,
            plain=True,
            algo=algo,
            perf=perf,
            dtype_messages=('fp16' if dtype == 'fp16' else 'fp32'),
            alpha=alpha,
            beta=None,
            device="cuda",
            seed=seed,
        )
        observable_decoder = _ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=True,
        )
    else:
        # Rust native MinSum BP decoder
        _RelayDecoderDyn_unused, _RustObservableRunner, _MinSumBPDecoderDyn = _select_backend("rust", dtype, require='plain')
        decoder = _MinSumBPDecoderDyn(
            check_matrices.check_matrix,
            error_priors=check_matrices.error_priors,
            max_iter=set_max_iter,
            alpha=None if (alpha == 0.0) else alpha,
            gamma0=None,
        )
        observable_decoder = _RustObservableRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=True,
        )

    total_shots = 0
    total_errors = 0
    iters_all = []
    # Decoder-only timing accumulators (nanoseconds)
    total_decode_ns = 0
    total_decode_shots = 0

    # Warmup decoder to avoid JIT/tuning overhead in timing measurements
    from relay_bp_triton.utils import warmup_build_and_decode
    warmup_build_and_decode(
        check_matrices=check_matrices,
        dem=dem,
        backend=backend,
        algo=algo,
        perf=perf,
        dtype=dtype,
        pre_iter=1,
        num_sets=0,
        set_max_iter=0,
        stop_nconv=1,
        gamma0=None,
        gamma_dist_interval=None,
        alpha=alpha,
        beta=None,
        device="cuda",
        seed=seed,
        repeats=3,
        batch=batch,
        parallel=parallel,
    )
    # Recreate sampler for the timed loop
    sampler = dem.compile_sampler(seed=seed)
    print(f"Collecting until {target_errors} errors or {max_shots} shots...")
    start_time = time.time()

    while total_errors < target_errors and total_shots < max_shots:
        # Sample errors and decode with detailed iteration tracking
        det, obs, errors = sampler.sample(batch, return_errors=True)

        # Decode with timing measurement
        t0_ns = time.perf_counter_ns()
        obs_det = observable_decoder.from_errors_decode_observables_detailed_batch(
            errors.astype(np.uint8), parallel=parallel
        )
        t1_ns = time.perf_counter_ns()
        total_decode_ns += (t1_ns - t0_ns)
        total_decode_shots += len(obs_det)
        
        # Update reproducibility hash
        _hash_update_arr(hasher, errors.astype(np.uint8, copy=False))
        O = check_matrices.observables_matrix.shape[0]
        pred_obs = np.empty((len(obs_det), O), dtype=np.uint8)
        for i, r in enumerate(obs_det):
            pred_obs[i, :] = np.asarray(r.observables, dtype=np.uint8)
        _hash_update_arr(hasher, pred_obs)
        
        # Extract iteration statistics
        iters_i32 = np.fromiter((r.iterations for r in obs_det), dtype=np.int32, count=len(obs_det))
        succ_arr  = np.fromiter((r.converged for r in obs_det), dtype=np.uint8, count=len(obs_det))
        fail_arr  = np.fromiter((r.error_detected for r in obs_det), dtype=np.uint8, count=len(obs_det))
        _hash_update_arr(hasher, iters_i32)
        _hash_update_arr(hasher, succ_arr)
        _hash_update_arr(hasher, fail_arr)
        iters_arr = np.array([r.iterations for r in obs_det], dtype=float)
        iters_all.extend(iters_arr.tolist())
        
        # Count logical errors and provide diagnostic breakdown
        errs = int(sum(r.error_detected for r in obs_det))
        n = len(obs_det)
        n_fail = errs
        n_conv = int(sum(r.converged for r in obs_det))
        n_ok_conv = int(sum((not r.error_detected) and r.converged for r in obs_det))
        n_bad_but_conv = int(sum(r.error_detected and r.converged for r in obs_det))
        n_bad_and_nconv = n_fail - n_bad_but_conv
        n_ok_and_nconv = int(sum((not r.error_detected) and (not r.converged) for r in obs_det))
        print(f"    batch breakdown: success&conv={n_ok_conv}, fail&conv={n_bad_but_conv}, fail&nconv={n_bad_and_nconv}, success&nconv={n_ok_and_nconv}, conv={n_conv}/{n}")
        total_errors += errs
        total_shots += batch
        print(f"  Collected {total_shots} shots, {total_errors} errors (batch errors: {errs})")

    ler = total_errors / total_shots if total_shots > 0 else 0.0
    per_cycle_ler = 1 - (1 - ler) ** (1 / rounds)

    obs_count = dem.num_observables
    # For CSS-style XZ decoding, obs_count ≈ 2*k; for single-basis Z or X, obs_count ≈ k.
    if basis.lower() == 'xz':
        k_guess = obs_count // 2
    else:
        k_guess = obs_count

    per_round_per_qubit_rate = ler / (rounds * max(1, k_guess))

    avg_bp_iterations = float(np.mean(iters_all)) if iters_all else 0.0

    runtime_per_shot = None
    decoder_runtime_per_shot = None
    decoder_runtime_per_iteration = None
    end_time = time.time()
    if start_time and end_time:
        wall_time_s = (end_time - start_time)
    if start_time and end_time and total_shots > 0:
        runtime_per_shot = (end_time - start_time) * 1e9 / total_shots
    if total_decode_shots > 0:
        decoder_runtime_per_shot = total_decode_ns / total_decode_shots
        total_iterations = float(np.sum(iters_all)) if iters_all else 0.0
        if total_iterations > 0:
            decoder_runtime_per_iteration = total_decode_ns / total_iterations

    out = {
        'shots': int(total_shots),
        'logical_errors': int(total_errors),
        'logical_error_rate': float(ler),
        'per_cycle_logical_error_rate': float(per_cycle_ler),
        'per_round_per_qubit_rate': float(per_round_per_qubit_rate),
        'avg_bp_iterations': float(avg_bp_iterations),
        # Aggregated timing totals
        'decoder_total_time_ns': int(total_decode_ns),
        'wall_time_s': (float(wall_time_s) if wall_time_s is not None else None),
        'config': {
            'circuit': str(circuit),
            'distance': int(distance),
            'rounds': int(rounds),
            'error_rate': float(error_rate),
            'set_max_iter': int(set_max_iter),
            'alpha': None if alpha is None else float(alpha),
            'batch': int(batch),
        }
    }
    if runtime_per_shot is not None:
        out['runtime_per_shot_ns'] = float(runtime_per_shot)
    if decoder_runtime_per_shot is not None:
        out['decoder_runtime_per_shot_ns'] = float(decoder_runtime_per_shot)
    if decoder_runtime_per_iteration is not None:
        out['decoder_runtime_per_iteration_ns'] = float(decoder_runtime_per_iteration)
    out['determinism_hash'] = hasher.hexdigest()
    return out


def main():
    """Main function for Relay-BP detailed analysis script."""
    args = parse_args()
    
    # Print experiment start information
    print("=" * 80)
    print("RELAY-BP DETAILED DECODER EXPERIMENT")
    print("=" * 80)
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Print all configuration options
    print("CONFIGURATION OPTIONS:")
    print("-" * 40)
    
    # Circuit parameters
    print("Circuit Parameters:")
    print(f"  Circuit: {args.circuit}")
    print(f"  Distance: {args.distance}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Error Rate: {args.error_rate}")
    print(f"  Basis: {args.basis}")
    print(f"  Mode: {args.mode}")
    print(f"  Perf: {args.perf}")
    print(f"  Backend: {args.backend or 'rust'}")
    print(f"  Dtype: {args.dtype}")
    print()

    if args.mode == 'relay':
        # Relay-BP parameters
        print("Relay-BP Parameters:")
        print(f"  Gamma0: {args.gamma0}")
        print(f"  Pre-iterations: {args.pre_iter}")
        print(f"  Number of Sets: {args.num_sets}")
        print(f"  Set Max Iterations: {args.set_max_iter}")
        print(f"  Gamma Distribution: [{args.gamma_dist_min}, {args.gamma_dist_max}]")
        print(f"  Stop N Converged: {args.stop_nconv}")
        print()
    else:
        # Plain-BP parameters
        print("Plain-BP Parameters:")
        print(f"  Set Max Iterations: {args.set_max_iter}")
        print(f"  Alpha (normalized min-sum): {args.alpha}")
        print()
    
    # Execution parameters
    print("Execution Parameters:")
    print(f"  Target Errors: {args.target_errors}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Max Shots: {args.max_shots}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Output Format: {args.output_format}")
    print(f"  Seed: {args.seed}")
    print()
    print("=" * 80)
    print()

    if args.mode == 'plain':
        results = run_plain_bp_detailed(args)
    elif args.mode == 'relay':
        results = run_relay_bp_detailed(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    # Output results based on format
    if args.output_format == 'simple':
        print(f"Shots: {results['shots']}, Logical Errors: {results['logical_errors']}")
        print(f"Logical Error Rate: {results['logical_error_rate']:.2e}")
        print(f"Per-Cycle Logical Error Rate: {results['per_cycle_logical_error_rate']:.2e}")
        legs_suffix = f" (legs: {results['avg_legs']:.1f})" if 'avg_legs' in results else ""
        print(f"Average BP Iterations: {results['avg_bp_iterations']:.1f}{legs_suffix}")
        if 'runtime_per_shot_ns' in results:
            print(f"Runtime per shot: {_format_time_units(results['runtime_per_shot_ns'])}")
        # Decoder-only timing metrics if available
        if 'decoder_runtime_per_shot_ns' in results:
            print(f"Decoder runtime per shot: {_format_time_units(results['decoder_runtime_per_shot_ns'])}")
        if 'decoder_runtime_per_iteration_ns' in results:
            print(f"Decoder runtime per iteration: {_format_time_units(results['decoder_runtime_per_iteration_ns'])}")
        if 'decoder_runtime_per_leg_ns' in results:
            print(f"Decoder runtime per leg: {_format_time_units(results['decoder_runtime_per_leg_ns'])}")
        print(f"Determinism hash: {results.get('determinism_hash','-')}")
    
    elif args.output_format == 'json':
        import json
        print(json.dumps(results, indent=2))
    
    elif args.output_format == 'csv':
        # Print CSV header and data
        if 'runtime_per_shot_ns' in results:
            print("shots,logical_errors,logical_error_rate,per_cycle_logical_error_rate,avg_bp_iterations,runtime_per_shot_ns,gamma0,num_sets,stop_nconv")
            print(f"{results['shots']},{results['logical_errors']},{results['logical_error_rate']:.6e},{results['per_cycle_logical_error_rate']:.6e},{results['avg_bp_iterations']:.1f},{results['runtime_per_shot_ns']:.1f},{results['config']['gamma0']},{results['config']['num_sets']},{results['config']['stop_nconv']}")
        else:
            print("shots,logical_errors,logical_error_rate,per_cycle_logical_error_rate,avg_bp_iterations,gamma0,num_sets,stop_nconv")
            print(f"{results['shots']},{results['logical_errors']},{results['logical_error_rate']:.6e},{results['per_cycle_logical_error_rate']:.6e},{results['avg_bp_iterations']:.1f},{results['config']['gamma0']},{results['config']['num_sets']},{results['config']['stop_nconv']}")


if __name__ == '__main__':
    main()
