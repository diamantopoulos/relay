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

import sys
import os

# Add the development source to the path to use the unified interface
# This MUST be done before any other imports that might import relay_bp
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import time
import numpy as np
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
from relay_bp.triton.utils import describe_check_matrices

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
    # Use the already imported relay_bp module
    _rb = relay_bp
    
    if backend == "triton":
        if dtype not in ("fp16", "fp32"):
            raise ValueError("Triton supports only dtype in {'fp16','fp32'}")
        
        # Use the new unified interface
        relay_cls = _rb.select_decoder(backend="triton", dtype=dtype, algorithm="relay")
        from relay_bp.triton.adapter import ObservableDecoderRunner as _ObservableDecoderRunner
        return relay_cls, _ObservableDecoderRunner, None

        
    elif backend == "rust":
        if dtype not in ("fp32", "fp64"):
            raise ValueError("Rust backend supports only dtype in {'fp32','fp64'} (no fp16 exports present)")
        
        relay_cls = None
        minsum_cls = None
        
        if require in (None, 'relay'):
            relay_cls = _rb.select_decoder(backend="rust", dtype=dtype, algorithm="relay")
        if require in (None, 'plain'):
            minsum_cls = _rb.select_decoder(backend="rust", dtype=dtype, algorithm="plain")

        from relay_bp import ObservableDecoderRunner as _ObservableDecoderRunner
        return relay_cls, _ObservableDecoderRunner, minsum_cls
    else:
        raise ValueError(f"Unknown backend: {backend}. Available: {_rb.get_available_backends()}")



def _format_time_from_us(us: float) -> str:
    """Format time in microseconds to a compact multi-unit string including ns."""
    ns = us * 1e3
    ms = us / 1e3
    s = us / 1e6
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




def run_relay_bp_experiment(circuit, basis, distance, rounds, error_rate, gamma0, pre_iter, 
                           num_sets, set_max_iter, gamma_dist_min, gamma_dist_max, stop_nconv,
                           target_errors=20, batch=2000, max_shots=1000000, parallel=True, seed=0,
                           backend: str | None = None, perf: str = "default", dtype: str = 'fp32'):
    """Deprecated: use run_bp_experiment_core with mode='relay'."""
    return run_bp_experiment_core(
        mode='relay',
        circuit=circuit,
        basis=basis,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
        gamma0=gamma0,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        gamma_dist_min=gamma_dist_min,
        gamma_dist_max=gamma_dist_max,
        stop_nconv=stop_nconv,
        alpha=None,
        target_errors=target_errors,
        batch=batch,
        max_shots=max_shots,
        parallel=parallel,
        seed=seed,
        backend=backend,
        perf=perf,
        dtype=dtype,
    )


## Deprecated wrappers (left intentionally empty after unification)


## Single entry point: use run_bp_experiment_core directly


def run_bp_experiment_core(
    *,
    mode: str,
    circuit, basis, distance, rounds, error_rate,
    # Relay params
    gamma0=None, pre_iter=0, num_sets=0, set_max_iter=0,
    gamma_dist_min=None, gamma_dist_max=None, stop_nconv=1,
    # Plain params
    alpha=None,
    # Common
    target_errors=20, batch=2000, max_shots=1000000, parallel=True, seed=0,
    backend: str | None = None, perf: str = "default", dtype: str = 'fp32',
):
    """Unified core experiment for Relay-BP (mode='relay') and Plain-BP (mode='plain')."""
    # Deterministic seeding using global seed
    random.seed(seed)
    np.random.seed(seed)

    # Get test circuit
    circuit_obj = get_test_circuit(
        circuit=circuit,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
    )

    # Apply basis filtering based on argument
    if basis == 'z':
        circuit_obj = filter_detectors_by_basis(circuit_obj, "Z")
    
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
    assert dem.num_observables > 0

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

    cfg = {
        "circuit": circuit,
        "basis": basis,
        "distance": distance,
        "rounds": rounds,
        "error_rate": error_rate,
        "parallel": bool(parallel),
        "stim_seed": seed,
        # Determinism-affecting runtime knobs
        "backend": backend,
        "perf": perf,
        "dtype": dtype,
        "batch": int(batch),
        "target_errors": int(target_errors),
        "max_shots": int(max_shots),
    }
    if mode == 'relay':
        cfg.update({
            "gamma0": gamma0, "pre_iter": pre_iter, "num_sets": num_sets,
            "set_max_iter": set_max_iter, "gamma_dist": [gamma_dist_min, gamma_dist_max],
            "stop_nconv": stop_nconv, "relay_seed": seed,
            # Record stopping criterion implicitly (nconv right now)
            "stopping_criterion": "nconv",
        })
    else:
        cfg.update({
            "set_max_iter": set_max_iter,
            "alpha": alpha,
        })
    hasher.update(json.dumps(cfg, sort_keys=True).encode())

    # Build decoder + runner
    if mode == 'relay':
        _RelayDecoderDyn, _ObservableDecoderRunner, _ = _select_backend(backend, dtype, require='relay')
        relay_kwargs = dict(
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
        if backend == 'triton':
            decoder = _RelayDecoderDyn(
                check_matrices.check_matrix,
                **relay_kwargs,
                algo=("relay" if num_sets > 0 else "plain"),
                perf=perf,
                dtype_messages=('fp16' if dtype == 'fp16' else 'fp32'),
            )
        else:
            decoder = _RelayDecoderDyn(check_matrices.check_matrix, **relay_kwargs)
        observable_decoder = _ObservableDecoderRunner(
            decoder,
            check_matrices.observables_matrix,
            include_decode_result=True,
        )
    else:
        if backend == "triton":
            from relay_bp.triton.adapter import RelayDecoder as _RelayDecoderFXX
            from relay_bp.triton.adapter import ObservableDecoderRunner as _ObservableDecoderRunner
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
                algo='plain',
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

    # Warmup (Triton only)
    if backend == 'triton':
        try:
            from relay_bp.triton.utils import warmup_build_and_decode
            if mode == 'relay':
                warmup_build_and_decode(
                    check_matrices=check_matrices, dem=dem, backend=backend, algo='relay', perf=perf, dtype=dtype,
                    pre_iter=1, num_sets=0, set_max_iter=0, stop_nconv=1,
                    gamma0=gamma0, gamma_dist_interval=(gamma_dist_min, gamma_dist_max),
                    alpha=None, beta=None, device="cuda", seed=seed, repeats=3, batch=batch, parallel=parallel,
                )
            else:
                warmup_build_and_decode(
                    check_matrices=check_matrices, dem=dem, backend=backend, algo='plain', perf=perf, dtype=dtype,
                    pre_iter=1, num_sets=0, set_max_iter=0, stop_nconv=1,
                    gamma0=None, gamma_dist_interval=None,
                    alpha=alpha, beta=None, device="cuda", seed=seed, repeats=3, batch=batch, parallel=parallel,
                )
        except Exception:
            pass

    # Stats accumulators
    total_shots = 0
    total_errors = 0
    bp_iters_all = []
    legs_all = []
    total_decode_ns = 0
    total_decode_shots = 0

    # Sampling loop
    sampler = dem.compile_sampler(seed=seed)
    print(f"Collecting until {target_errors} errors or {max_shots} shots...")
    start_time = time.time()
    while total_errors < target_errors and total_shots < max_shots:
        det, obs, errors = sampler.sample(batch, return_errors=True)
        t0_ns = time.perf_counter_ns()
        obs_det = observable_decoder.from_errors_decode_observables_detailed_batch(
            errors.astype(np.uint8), parallel=parallel
        )
        t1_ns = time.perf_counter_ns()
        total_decode_ns += (t1_ns - t0_ns)
        total_decode_shots += len(obs_det)

        # Hash updates
        _hash_update_arr(hasher, errors.astype(np.uint8, copy=False))
        O = check_matrices.observables_matrix.shape[0]
        pred_obs = np.empty((len(obs_det), O), dtype=np.uint8)
        for i, r in enumerate(obs_det):
            pred_obs[i, :] = np.asarray(r.observables, dtype=np.uint8)
        _hash_update_arr(hasher, pred_obs)

        # Stats
        iters_arr = np.array([r.iterations for r in obs_det], dtype=float)
        iters_i32 = np.fromiter((r.iterations for r in obs_det), dtype=np.int32, count=len(obs_det))
        succ_arr  = np.fromiter((r.converged for r in obs_det), dtype=np.uint8, count=len(obs_det))
        fail_arr  = np.fromiter((r.error_detected for r in obs_det), dtype=np.uint8, count=len(obs_det))
        _hash_update_arr(hasher, iters_i32)
        _hash_update_arr(hasher, succ_arr)
        _hash_update_arr(hasher, fail_arr)

        bp_iters_all.extend(iters_arr.tolist())
        if mode == 'relay' and set_max_iter > 0:
            legs_all.extend(1.0 + np.maximum(0.0, (iters_arr - pre_iter) / set_max_iter))

        errs = int(sum(r.error_detected for r in obs_det))
        n = len(obs_det)
        n_conv = int(sum(r.converged for r in obs_det))
        n_ok = int(sum((not r.error_detected) and r.converged for r in obs_det))
        n_bad_but_conv = int(sum(r.error_detected and r.converged for r in obs_det))
        n_bad_and_nconv = errs - n_bad_but_conv
        n_ok_and_nconv = int(sum((not r.error_detected) and (not r.converged) for r in obs_det))
        print(f"    batch breakdown: success&conv={n_ok}, fail&conv={n_bad_but_conv}, fail&nconv={n_bad_and_nconv}, success&nconv={n_ok_and_nconv}, conv={n_conv}/{n}")
        total_errors += errs
        total_shots += batch
        print(f"  Collected {total_shots} shots, {total_errors} errors (batch errors: {errs})")

    # Rates and averages
    logical_error_rate = total_errors / total_shots if total_shots > 0 else 0.0
    per_cycle_logical_error_rate = 1 - (1 - logical_error_rate) ** (1 / rounds)

    obs_count = dem.num_observables
    if basis.lower() == 'xz':
        k_guess = obs_count // 2
    else:
        k_guess = obs_count
    per_round_per_qubit_rate = logical_error_rate / (rounds * max(1, k_guess))

    if bp_iters_all:
        avg_bp_iterations = float(np.mean(bp_iters_all))
        avg_legs = float(np.mean(legs_all)) if (mode == 'relay' and legs_all) else 0.0
    else:
        avg_bp_iterations = 0.0
        avg_legs = 0.0

    # Timing
    end_time = time.time()
    results = {
        'shots': int(total_shots),
        'logical_errors': int(total_errors),
        'logical_error_rate': float(logical_error_rate),
        'per_cycle_logical_error_rate': float(per_cycle_logical_error_rate),
        'per_round_per_qubit_rate': float(per_round_per_qubit_rate),
        'avg_bp_iterations': float(avg_bp_iterations),
        'config': {
            'circuit': str(circuit), 'distance': int(distance), 'rounds': int(rounds),
            'error_rate': float(error_rate), 'batch': int(batch),
        },
    }
    if mode == 'relay':
        results['avg_legs'] = float(avg_legs)
        results['config'].update({
            'gamma0': float(gamma0), 'num_sets': int(num_sets), 'stop_nconv': int(stop_nconv),
            'gamma_dist_interval': (float(gamma_dist_min), float(gamma_dist_max)),
        })

    # Per-shot and totals
    if start_time and end_time and total_shots > 0:
        total_runtime_seconds = end_time - start_time
        results['runtime_per_shot_us'] = float((total_runtime_seconds * 1e6) / total_shots)
        results['total_runtime_us'] = float(total_runtime_seconds * 1e6)
    if total_decode_shots > 0:
        total_iterations = float(np.sum(bp_iters_all)) if bp_iters_all else 0.0
        # convert decoder times to microseconds
        results['decoder_total_time_us'] = float(total_decode_ns) / 1e3
        results['decoder_runtime_per_shot_us'] = (float(total_decode_ns) / 1e3) / float(total_decode_shots)
        if total_iterations > 0:
            results['decoder_runtime_per_iteration_us'] = (float(total_decode_ns) / 1e3) / float(total_iterations)
        # expose totals for downstream throughput computation
        results['total_decode_shots'] = int(total_decode_shots)
        results['total_iterations'] = float(total_iterations)
        if mode == 'relay' and legs_all:
            total_legs = float(np.sum(legs_all))
            if total_legs > 0:
                results['decoder_runtime_per_leg_us'] = (float(total_decode_ns) / 1e3) / float(total_legs)

    # Attach diagnostic
    results['per_cycle_ler_alt_half_rounds'] = float(1 - (1 - logical_error_rate) ** (1 / max(1, 2 * rounds)))
    results['determinism_hash'] = hasher.hexdigest()
    return results

def run_plain_bp_experiment(circuit, basis, distance, rounds, error_rate,
                            set_max_iter=200, alpha=None, target_errors=20, batch=2000,
                            max_shots=1000000, parallel=True, seed=0,
                            backend: str | None = None, perf: str = "default", algo: str = "plain",
                            dtype: str = 'fp16'):
    """Deprecated: use run_bp_experiment_core with mode='plain'."""
    return run_bp_experiment_core(
        mode='plain',
        circuit=circuit,
        basis=basis,
        distance=distance,
        rounds=rounds,
        error_rate=error_rate,
        gamma0=None,
        pre_iter=0,
        num_sets=0,
        set_max_iter=set_max_iter,
        gamma_dist_min=None,
        gamma_dist_max=None,
        stop_nconv=1,
        alpha=alpha,
        target_errors=target_errors,
        batch=batch,
        max_shots=max_shots,
        parallel=parallel,
        seed=seed,
        backend=backend,
        perf=perf,
        dtype=dtype,
    )


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

    results = run_bp_experiment_core(
        mode=args.mode,
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
    
    # Output results based on format
    if args.output_format == 'simple':
        print(f"Shots: {results['shots']}, Logical Errors: {results['logical_errors']}")
        print(f"Logical Error Rate: {results['logical_error_rate']:.2e}")
        print(f"Per-Cycle Logical Error Rate: {results['per_cycle_logical_error_rate']:.2e}")
        legs_suffix = f" (legs: {results['avg_legs']:.1f})" if 'avg_legs' in results else ""
        print(f"Average BP Iterations: {results['avg_bp_iterations']:.1f}{legs_suffix}")
        if 'runtime_per_shot_us' in results:
            print(f"Runtime per shot: {_format_time_from_us(results['runtime_per_shot_us'])}")
        # Decoder-only timing metrics if available
        if 'decoder_runtime_per_shot_us' in results:
            print(f"Decoder runtime per shot: {_format_time_from_us(results['decoder_runtime_per_shot_us'])}")
        if 'decoder_runtime_per_iteration_us' in results:
            print(f"Decoder runtime per iteration: {_format_time_from_us(results['decoder_runtime_per_iteration_us'])}")
        if 'decoder_runtime_per_leg_us' in results:
            print(f"Decoder runtime per leg: {_format_time_from_us(results['decoder_runtime_per_leg_us'])}")
        if 'total_runtime_us' in results:
            print(f"Total runtime (Monte Carlo): {_format_time_from_us(results['total_runtime_us'])}")
        if 'decoder_total_time_us' in results:
            print(f"Decoder total time: {_format_time_from_us(results['decoder_total_time_us'])}")
        # Throughput metrics
        wall_time_us = results.get('total_runtime_us')
        shots_val = results.get('shots')
        if wall_time_us is not None and shots_val is not None:
            wall_time_s = float(wall_time_us) / 1e6
            shots = float(shots_val)
            if wall_time_s > 0.0 and shots > 0.0:
                print(f"Shots per second (end-to-end): {shots / wall_time_s:.2f} shots/s")

        dec_time_us_val = results.get('decoder_total_time_us')
        total_decode_shots_val = results.get('total_decode_shots')
        total_iterations_val = results.get('total_iterations')
        if dec_time_us_val is not None and float(dec_time_us_val) > 0.0:
            dec_time_s = float(dec_time_us_val) / 1e6
            if total_decode_shots_val is not None and float(total_decode_shots_val) > 0.0:
                print(f"Decoder shots per second: {float(total_decode_shots_val) / dec_time_s:.2f} shots/s")
            if total_iterations_val is not None and float(total_iterations_val) > 0.0:
                print(f"Iterations per second: {float(total_iterations_val) / dec_time_s:.2f} it/s")
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
