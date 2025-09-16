#!/usr/bin/env python3
"""
Relay-BP Detailed Decoder Script

This script directly uses Relay-BP decoder to get iteration counts and per-cycle error rates
matching the paper's methodology.
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Relay-BP decoder with detailed iteration counting')
    
    # Circuit parameters
    parser.add_argument('--circuit', type=str, default='bicycle_bivariate_144_12_12_memory_choi_XZ',
                       help='Circuit name (default: bicycle_bivariate_144_12_12_memory_Z)')
    parser.add_argument('--basis', type=str, choices=['z', 'xz'], default='xz', help='Decoding basis (default: xz)')
    parser.add_argument('--distance', type=int, default=12, help='Code distance (default: 12)')
    parser.add_argument('--rounds', type=int, default=12, help='Number of rounds (default: 12)')
    parser.add_argument('--error-rate', type=float, default=0.003, help='Physical error rate (default: 0.003)')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots (default: 1000)')
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
    
    # Execution parameters
    parser.add_argument('--parallel', action='store_true', help='Enable Relay-BP builtin parallelism (default: False)')
    parser.add_argument('--output-format', type=str, choices=['simple', 'json', 'csv'], default='simple',
                       help='Output format (default: simple)')
    parser.add_argument('--measure-time', action='store_true', help='Measure and report execution time')
    
    return parser.parse_args()


def run_relay_bp_detailed(args):
    """Run Relay-BP with detailed iteration counting."""
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
        measure_time=args.measure_time
    )


def run_relay_bp_experiment(circuit, basis, distance, rounds, error_rate, gamma0, pre_iter, 
                           num_sets, set_max_iter, gamma_dist_min, gamma_dist_max, stop_nconv,
                           target_errors=20, batch=2000, max_shots=1000000, parallel=True, 
                           measure_time=True):
    """Run Relay-BP experiment with detailed iteration counting - reusable function."""
    
    # Deterministic seeding (optional but helpful for reproducibility)
    try:
        random.seed(0)
        np.random.seed(0)
        stim.set_global_seed(0)
    except Exception:
        pass

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
    
    # Check observables count
    print(f"DEM stats: detectors={dem.num_detectors}, observables={dem.num_observables}")
    print(f"Using rounds={rounds} as the number of cycles for per-cycle LER conversion")
    # Soft filename sanity check that circuit tag encodes distance/rounds
    if isinstance(circuit, str):
        tag_ok = f"_{rounds}_" in circuit
        if not tag_ok:
            print(f"[warn] Circuit name '{circuit}' does not obviously encode rounds={rounds}.")
    assert dem.num_observables > 0, (
        "No observables after basis filtering. "
        "Use XZ (no filter) to match the paper."
    )
    
    # Create check matrices
    check_matrices = CheckMatrices.from_dem(dem)
    
    # Create Relay-BP decoder (CPU). Use normalized min-sum alpha if provided.
    decoder = relay_bp.RelayDecoderF64(
        check_matrices.check_matrix,
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
    
    # Create observable decoder
    observable_decoder = relay_bp.ObservableDecoderRunner(
        decoder,
        check_matrices.observables_matrix,
        include_decode_result=True,  # Enable detailed results
    )
    
    # BP iteration stats (paper's x-axis) and debug legs reporting
    avg_bp_iterations = 0.0
    avg_legs = 0.0
    legs_all = []
    # Error-targeted collection (prevents LER=0 and stabilizes estimates)
    # Apply heuristic to avoid stopping too early at low LER
    heuristic_max = 100 * target_errors * rounds
    max_shots = max(max_shots, heuristic_max)
    
    total_shots = 0
    total_errors = 0
    bp_iters_all = []
    
    
    sampler = circuit_obj.compile_detector_sampler()
    print(f"Collecting until {target_errors} errors or {max_shots} shots...")

    # Measure time around the actual error-targeted collection loop
    start_time = time.time() if measure_time else None
    
    while total_errors < target_errors and total_shots < max_shots:
        synd, obs = sampler.sample(batch, separate_observables=True)
        synd_u8 = synd.astype(np.uint8)
        
        # Get detailed results for iteration counting
        det = observable_decoder.decode_detailed_batch(synd_u8, parallel=parallel)
        iters_arr = np.array([r.iterations for r in det], dtype=float)
        bp_iters_all.extend(iters_arr.tolist())  # r.iterations = BP iterations
        # Derive legs only for reporting/debug (not for avg iteration computation)
        legs_all.extend(1.0 + np.maximum(0.0, (iters_arr - pre_iter) / set_max_iter))
        
        # Get predictions for error counting using detailed path to ensure any frames are applied
        obs_det = observable_decoder.decode_observables_detailed_batch(synd_u8, parallel=parallel)
        # Extract observables from results list into array
        pred = np.stack([r.observables for r in obs_det], axis=0)
        pred = pred.astype(np.uint8, copy=False)
        
        # Count errors
        assert pred.shape == obs.shape and pred.shape[1] > 0, \
            f"Bad shapes: pred{pred.shape} vs obs{obs.shape}"
        obs_u8 = obs.astype(np.uint8, copy=False)
        # Count failure if any logical observable differs (mod 2)
        errs = (np.bitwise_xor(pred, obs_u8).any(axis=1)).sum()
        total_errors += int(errs)
        total_shots += batch
        
        print(f"  Collected {total_shots} shots, {total_errors} errors (batch errors: {errs})")
        
        # Debug: Show sample predictions vs expected
        if total_shots <= 2000:  # Only for first batch
            print(f"    Sample pred: {pred[0]}")
            print(f"    Sample obs:  {obs[0].astype(int)}")
            print(f"    Sample diff: {pred[0] != obs[0].astype(int)}")
    
    logical_error_rate = total_errors / total_shots if total_shots > 0 else 0.0
    
    # Calculate per-cycle logical error rate (as per paper)
    per_cycle_logical_error_rate = 1 - (1 - logical_error_rate) ** (1 / rounds)
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
    
    # Calculate runtime per shot
    runtime_per_shot = None
    end_time = time.time() if measure_time else None
    if measure_time and start_time and end_time and total_shots > 0:
        total_runtime_seconds = end_time - start_time
        runtime_per_shot = (total_runtime_seconds * 1e9) / total_shots  # Convert to nanoseconds
    
    # Prepare results (convert numpy types to Python types for JSON serialization)
    results_data = {
        'shots': int(total_shots),
        'logical_errors': int(total_errors),
        'logical_error_rate': float(logical_error_rate),
        'per_cycle_logical_error_rate': float(per_cycle_logical_error_rate),
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
        }
    }
    
    if runtime_per_shot is not None:
        results_data['runtime_per_shot_ns'] = float(runtime_per_shot)
    # Attach optional diagnostic to help verify cycle root choice
    results_data['per_cycle_ler_alt_half_rounds'] = float(per_cycle_ler_half_rounds)
    
    return results_data


def run_plain_bp_experiment(circuit, basis, distance, rounds, error_rate,
                            max_iter=200, alpha=None, target_errors=20, batch=2000,
                            max_shots=1000000, parallel=True, measure_time=True):
    """Run plain min-sum BP (no relay, no memory) with detailed counting."""

    try:
        random.seed(0)
        np.random.seed(0)
        stim.set_global_seed(0)
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

    # Build plain min-sum decoder (no relay, no memory)
    kwargs = dict(
        error_priors=check_matrices.error_priors,
        max_iter=max_iter,
        alpha=None if (alpha == 0.0) else alpha,
        gamma0=None,
    )
    decoder = relay_bp.MinSumBPDecoderF64(check_matrices.check_matrix, **kwargs)
    observable_decoder = relay_bp.ObservableDecoderRunner(
        decoder,
        check_matrices.observables_matrix,
        include_decode_result=True,
    )

    total_shots = 0
    total_errors = 0
    iters_all = []

    sampler = circuit_obj.compile_detector_sampler()
    print(f"Collecting until {target_errors} errors or {max_shots} shots...")
    start_time = time.time() if measure_time else None

    while total_errors < target_errors and total_shots < max_shots:
        synd, obs = sampler.sample(batch, separate_observables=True)
        synd_u8 = synd.astype(np.uint8)

        # Detailed decoding to get iterations and predicted observables
        det = observable_decoder.decode_detailed_batch(synd_u8, parallel=parallel)
        iters_arr = np.array([r.iterations for r in det], dtype=float)
        iters_all.extend(iters_arr.tolist())

        obs_det = observable_decoder.decode_observables_detailed_batch(synd_u8, parallel=parallel)
        pred = np.stack([r.observables for r in obs_det], axis=0).astype(np.uint8, copy=False)
        obs_u8 = obs.astype(np.uint8, copy=False)
        errs = (np.bitwise_xor(pred, obs_u8).any(axis=1)).sum()
        total_errors += int(errs)
        total_shots += batch
        print(f"  Collected {total_shots} shots, {total_errors} errors (batch errors: {errs})")

    ler = total_errors / total_shots if total_shots > 0 else 0.0
    per_cycle_ler = 1 - (1 - ler) ** (1 / rounds)
    avg_bp_iterations = float(np.mean(iters_all)) if iters_all else 0.0

    runtime_per_shot = None
    end_time = time.time() if measure_time else None
    if measure_time and start_time and end_time and total_shots > 0:
        runtime_per_shot = (end_time - start_time) * 1e9 / total_shots

    out = {
        'shots': int(total_shots),
        'logical_errors': int(total_errors),
        'logical_error_rate': float(ler),
        'per_cycle_logical_error_rate': float(per_cycle_ler),
        'avg_bp_iterations': float(avg_bp_iterations),
        'config': {
            'circuit': str(circuit),
            'distance': int(distance),
            'rounds': int(rounds),
            'error_rate': float(error_rate),
            'max_iter': int(max_iter),
            'alpha': None if alpha is None else float(alpha),
        }
    }
    if runtime_per_shot is not None:
        out['runtime_per_shot_ns'] = float(runtime_per_shot)
    return out


def main():
    """Main function."""
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
    print()
    
    # Relay-BP parameters
    print("Relay-BP Parameters:")
    print(f"  Gamma0: {args.gamma0}")
    print(f"  Pre-iterations: {args.pre_iter}")
    print(f"  Number of Sets: {args.num_sets}")
    print(f"  Set Max Iterations: {args.set_max_iter}")
    print(f"  Gamma Distribution: [{args.gamma_dist_min}, {args.gamma_dist_max}]")
    print(f"  Stop N Converged: {args.stop_nconv}")
    print()
    
    # Execution parameters
    print("Execution Parameters:")
    print(f"  Shots: {args.shots}")
    print(f"  Target Errors: {args.target_errors}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Max Shots: {args.max_shots}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Output Format: {args.output_format}")
    print(f"  Measure Time: {args.measure_time}")
    print()
    print("=" * 80)
    print()
    
    # Run Relay-BP with detailed counting
    results = run_relay_bp_detailed(args)
    
    # Output results based on format
    if args.output_format == 'simple':
        print(f"Shots: {results['shots']}, Logical Errors: {results['logical_errors']}")
        print(f"Logical Error Rate: {results['logical_error_rate']:.2e}")
        print(f"Per-Cycle Logical Error Rate: {results['per_cycle_logical_error_rate']:.2e}")
        print(f"Average BP Iterations: {results['avg_bp_iterations']:.1f} (legs: {results['avg_legs']:.1f})")
        if 'runtime_per_shot_ns' in results:
            print(f"Runtime per shot: {results['runtime_per_shot_ns']:.1f} ns")
    
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
