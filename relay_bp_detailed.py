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
sys.path.append(str(Path(__file__).parent / "tests"))
from testdata import get_test_circuit, filter_detectors_by_basis

import stim
import relay_bp
from relay_bp.stim.sinter.check_matrices import CheckMatrices


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Relay-BP decoder with detailed iteration counting')
    
    # Circuit parameters
    parser.add_argument('--circuit', type=str, default='bicycle_bivariate_144_12_12_memory_Z',
                       help='Circuit name (default: bicycle_bivariate_144_12_12_memory_Z)')
    parser.add_argument('--distance', type=int, default=12, help='Code distance (default: 12)')
    parser.add_argument('--rounds', type=int, default=12, help='Number of rounds (default: 12)')
    parser.add_argument('--error-rate', type=float, default=0.003, help='Physical error rate (default: 0.003)')
    parser.add_argument('--shots', type=int, default=1000, help='Number of shots (default: 1000)')
    parser.add_argument('--basis', type=str, choices=['z', 'xz'], default='xz', help='Decoding basis (default: xz)')
    parser.add_argument('--target-errors', type=int, default=200, help='Target number of errors to collect (default: 200)')
    parser.add_argument('--batch', type=int, default=2000, help='Batch size for error collection (default: 2000)')
    parser.add_argument('--max-shots', type=int, default=1000000, help='Maximum shots to collect (default: 1000000)')
    
    # Relay-BP parameters
    parser.add_argument('--gamma0', type=float, default=0.125, help='Ordered memory parameter (default: 0.125)')
    parser.add_argument('--pre-iter', type=int, default=80, help='Pre-iterations (default: 80)')
    parser.add_argument('--num-sets', type=int, default=301, help='Number of Relay-BP sets (default: 301)')
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
    
    # Deterministic seeding (optional but helpful for reproducibility)
    try:
        random.seed(0)
        np.random.seed(0)
        stim.set_global_seed(0)
    except Exception:
        pass

    # Get test circuit
    circuit = get_test_circuit(
        circuit=args.circuit, 
        distance=args.distance, 
        rounds=args.rounds, 
        error_rate=args.error_rate
    )
    
    # Apply basis filtering based on argument
    if args.basis == 'z':
        circuit = filter_detectors_by_basis(circuit, "Z")
    # For XZ decoding, don't filter (keep all observables)
    
    # Create detector error model
    dem = circuit.detector_error_model()
    
    # Check observables count
    print(f"DEM stats: detectors={dem.num_detectors}, observables={dem.num_observables}")
    assert dem.num_observables > 0, (
        "No observables after basis filtering. "
        "Use XZ (no filter) to match the paper."
    )
    
    # Create check matrices
    check_matrices = CheckMatrices.from_dem(dem)
    
    # Create Relay-BP decoder
    decoder = relay_bp.RelayDecoderF64(
        check_matrices.check_matrix,
        error_priors=check_matrices.error_priors,
        gamma0=args.gamma0,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        gamma_dist_interval=(args.gamma_dist_min, args.gamma_dist_max),
        stop_nconv=args.stop_nconv,
        stopping_criterion="nconv",
        logging=False,
    )
    
    # Create observable decoder
    observable_decoder = relay_bp.ObservableDecoderRunner(
        decoder,
        check_matrices.observables_matrix,
        include_decode_result=True,  # Enable detailed results
    )
    
    # Convert legs → BP iterations (paper's x-axis) - will be updated after error collection
    avg_legs = 0.0  # Will be updated
    avg_bp_iterations = 0.0  # Will be updated
    
    # Error-targeted collection (prevents LER=0 and stabilizes estimates)
    target_errors = args.target_errors
    batch = args.batch
    # Apply heuristic to avoid stopping too early at low LER
    heuristic_max = 100 * target_errors * args.rounds
    max_shots = max(args.max_shots, heuristic_max)
    
    total_shots = 0
    total_errors = 0
    legs_all = []
    
    sampler = circuit.compile_detector_sampler()
    print(f"Collecting until {target_errors} errors or {max_shots} shots...")

    # Measure time around the actual error-targeted collection loop
    start_time = time.time() if args.measure_time else None
    
    while total_errors < target_errors and total_shots < max_shots:
        synd, obs = sampler.sample(batch, separate_observables=True)
        synd_u8 = synd.astype(np.uint8)
        
        # Get detailed results for iteration counting
        det = observable_decoder.decode_detailed_batch(synd_u8, parallel=args.parallel)
        legs_all.extend([r.iterations for r in det])
        
        # Get predictions for error counting
        pred = observable_decoder.decode_observables_batch(synd_u8, parallel=args.parallel)
        
        # Count errors
        assert pred.shape == obs.shape and pred.shape[1] > 0, \
            f"Bad shapes: pred{pred.shape} vs obs{obs.shape}"
        errs = (pred != obs.astype(int)).any(axis=1).sum()
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
    per_cycle_logical_error_rate = 1 - (1 - logical_error_rate) ** (1 / args.rounds)
    
    # Update BP iterations calculation with collected data
    if legs_all:
        legs = np.asarray(legs_all, dtype=float)
        avg_legs = float(legs.mean())
        T0 = args.pre_iter  # 80
        Tr = args.set_max_iter  # 60
        avg_bp_iterations = T0 + max(0.0, avg_legs - 1.0) * Tr
    else:
        avg_legs = 0.0
        avg_bp_iterations = 0.0
    
    # Calculate runtime per shot
    runtime_per_shot = None
    end_time = time.time() if args.measure_time else None
    if args.measure_time and start_time and end_time and total_shots > 0:
        total_runtime_seconds = end_time - start_time
        runtime_per_shot = (total_runtime_seconds * 1e9) / total_shots  # Convert to nanoseconds
    
    # Prepare results (convert numpy types to Python types for JSON serialization)
    results_data = {
        'shots': int(total_shots),
        'logical_errors': int(total_errors),
        'logical_error_rate': float(logical_error_rate),
        'per_cycle_logical_error_rate': float(per_cycle_logical_error_rate),
        'avg_bp_iterations': float(avg_bp_iterations),  # Paper's x-axis
        'avg_legs': float(avg_legs),  # For debugging
        'config': {
            'gamma0': float(args.gamma0),
            'num_sets': int(args.num_sets),
            'stop_nconv': int(args.stop_nconv),
            'gamma_dist_interval': (float(args.gamma_dist_min), float(args.gamma_dist_max)),
            'circuit': str(args.circuit),
            'distance': int(args.distance),
            'rounds': int(args.rounds),
            'error_rate': float(args.error_rate),
        }
    }
    
    if runtime_per_shot is not None:
        results_data['runtime_per_shot_ns'] = float(runtime_per_shot)
    
    return results_data


def main():
    """Main function."""
    args = parse_args()
    
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
