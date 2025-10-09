#!/usr/bin/env python3
"""
Relay-BP Paper Study Script

This script replicates the paper's methodology exactly for comprehensive benchmarking
of Relay-BP vs Plain-BP performance across different parameter configurations.

Paper methodology:
- X-axis: Average BP iteration count (not wall time)
- Y-axis: Per-cycle logical error rate (not per-shot)
- Parameters: S = solutions sought, R = max relay legs

The script performs systematic parameter sweeps to generate performance curves
matching the paper's analysis, supporting both Rust and Triton backends with
comprehensive result tracking and export capabilities.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import csv

# Import the reusable function from relay_bp_detailed
from relay_bp_detailed import run_relay_bp_experiment, run_plain_bp_experiment


class RelayBPPaperStudy:
    """Study class for replicating the paper's methodology with systematic parameter sweeps.
    
    This class implements comprehensive benchmarking of Relay-BP vs Plain-BP performance
    across different parameter configurations, generating performance curves that match
    the paper's analysis methodology.
    
    Features:
    - Systematic parameter sweeps (S, R, iteration limits)
    - Multiple backend support (Rust/Triton)
    - Incremental result saving for robustness
    - Comprehensive CSV export with all metrics
    """
    
    def __init__(self, output_dir: str = "paper_study_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def _csv_fieldnames(self) -> list[str]:
        """Define CSV field names for comprehensive result export."""
        return [
            # Configuration parameters
            'config_name', 'algo', 'perf', 'backend', 'dtype', 'batch',
            'num_sets', 'stop_nconv', 'gamma0', 'gamma_dist_min', 'gamma_dist_max',
            'pre_iter', 'set_max_iter', 'circuit', 'distance', 'rounds', 'error_rate', 'basis',
            # Experiment results
            'shots', 'logical_errors', 'logical_error_rate', 'per_cycle_logical_error_rate',
            'per_round_per_qubit_rate', 'avg_bp_iterations', 'avg_legs',
            'runtime_per_shot_ns', 'decoder_runtime_per_shot_ns', 'decoder_runtime_per_iteration_ns', 'decoder_runtime_per_leg_ns',
            'decoder_total_time_ns', 'wall_time_s'
        ]

    def _row_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert experiment result to CSV row format."""
        return {
            'config_name': result.get('config_name', ''),
            'algo': result.get('algo', ''),
            'perf': result.get('perf', ''),
            'backend': result.get('backend', ''),
            'dtype': result.get('dtype', ''),
            'batch': result.get('config', {}).get('batch', ''),
            'num_sets': result.get('num_sets', ''),
            'stop_nconv': result.get('stop_nconv', ''),
            'gamma0': result.get('gamma0', ''),
            'gamma_dist_min': result.get('gamma_dist_interval', [None, None])[0] if result.get('gamma_dist_interval') else '',
            'gamma_dist_max': result.get('gamma_dist_interval', [None, None])[1] if result.get('gamma_dist_interval') else '',
            'pre_iter': result.get('pre_iter', ''),
            'set_max_iter': result.get('set_max_iter', ''),
            'circuit': result.get('config', {}).get('circuit', ''),
            'distance': result.get('config', {}).get('distance', ''),
            'rounds': result.get('config', {}).get('rounds', ''),
            'error_rate': result.get('config', {}).get('error_rate', ''),
            'basis': 'xz',
            'shots': result.get('shots', ''),
            'logical_errors': result.get('logical_errors', ''),
            'logical_error_rate': result.get('logical_error_rate', ''),
            'per_cycle_logical_error_rate': result.get('per_cycle_logical_error_rate', ''),
            'per_round_per_qubit_rate': result.get('per_round_per_qubit_rate', ''),
            'avg_bp_iterations': result.get('avg_bp_iterations', ''),
            'avg_legs': result.get('avg_legs', ''),
            'runtime_per_shot_ns': result.get('runtime_per_shot_ns', ''),
            'decoder_runtime_per_shot_ns': result.get('decoder_runtime_per_shot_ns', ''),
            'decoder_runtime_per_iteration_ns': result.get('decoder_runtime_per_iteration_ns', ''),
            'decoder_runtime_per_leg_ns': result.get('decoder_runtime_per_leg_ns', ''),
            'decoder_total_time_ns': result.get('decoder_total_time_ns', ''),
            'wall_time_s': result.get('wall_time_s', ''),
        }

    def save_result_incremental(self, result: Dict[str, Any]):
        """Append a single result to CSV immediately for robustness during long runs.
        
        This method saves results incrementally to prevent data loss during
        long-running parameter sweeps, creating both CSV and JSON backups.
        """
        csv_path = self.output_dir / "paper_study_results.csv"
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self._csv_fieldnames())
            if write_header:
                writer.writeheader()
            writer.writerow(self._row_from_result(result))
        
        # Save JSON backup for robustness
        json_path = self.output_dir / f"{result.get('config_name','result')}.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

    
    def define_configs(self) -> List[Dict[str, Any]]:
        """Build systematic parameter sweep configurations for paper replication.
        
        This method defines the parameter space for comprehensive benchmarking,
        including Plain-BP iteration sweeps and Relay-BP parameter grids that
        match the paper's methodology.
        
        Returns:
            List of configuration dictionaries for systematic parameter sweeps
        """
        configs: List[Dict[str, Any]] = []
        backends = ['rust', 'triton']
        batch_values = [1]  # Single batch for detailed analysis

        # Plain BP sweep: vary max iterations to generate performance curves
        set_max_iter_values = [1, 5, 10, 20, 40, 60, 80, 100, 200, 300, 500, 600, 700, 1000, 1500, 2000, 5000, 10000]
        for backend in backends:
            dtypes = ['fp32']  # Focus on fp32 for consistency
            for dtype in dtypes:
                for tmax in set_max_iter_values:
                    for batch in batch_values:
                        configs.append({
                            'name': f'PlainBP-maxiter{tmax}-B{batch}-{backend}-{dtype}',
                            'algo': 'plain',
                            'perf': 'throughput',
                            'backend': backend,
                            'dtype': dtype,
                            'batch': batch,
                            'set_max_iter': tmax,
                            'alpha': None,
                        })

        # Relay-BP sweep: S (stop_nconv) and R (num_sets) parameter grid
        # Uncomment to enable Relay-BP parameter sweeps
        # stop_nconv_values = [1, 2, 3, 5, 7, 9]
        # num_sets_values = [1, 3, 5, 9, 13, 21, 45]
        # for backend in backends:
        #     for s in stop_nconv_values:
        #         for r in num_sets_values:
        #             configs.append({
        #                 'name': f'Relay-BP-S{s}-R{r}-{backend}',
        #                 'algo': 'relay',
        #                 'perf': ('throughput' if backend == 'triton' else 'default'),
        #                 'backend': backend,
        #                 'num_sets': r,
        #                 'gamma0': 0.125,
        #                 'gamma_dist_interval': (-0.24, 0.66),
        #                 'pre_iter': 80,
        #                 'set_max_iter': 60,
        #                 'stop_nconv': s,
        #             })

        return configs

    def run_any_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single configuration and return results with metadata.
        
        This method executes either Plain-BP or Relay-BP experiments based on
        the configuration, using the paper's standard circuit parameters.
        
        Args:
            config: Configuration dictionary with algorithm parameters
            
        Returns:
            Experiment results dictionary with performance metrics
        """
        print(f"Running {config['name']}...")
        algo = config.get('algo', 'relay')
        perf = config.get('perf', 'default')
        backend = config.get('backend', None)
        dtype = config.get('dtype', 'fp32')
        
        try:
            if algo == 'plain':
                out = run_plain_bp_experiment(
                    circuit='bicycle_bivariate_144_12_12_memory_choi_XZ',
                    basis='xz',
                    distance=12,
                    rounds=12,
                    error_rate=0.003,
                    set_max_iter=config['set_max_iter'],
                    alpha=config.get('alpha', None),
                    target_errors=20,
                    batch=config.get('batch', 2048),
                    max_shots=1,
                    parallel=True,
                    backend=backend,
                    perf=perf,
                    dtype=dtype,
                )
                out['algo'] = 'plain'
                out['set_max_iter'] = config['set_max_iter']
            else:
                out = run_relay_bp_experiment(
                    circuit='bicycle_bivariate_144_12_12_memory_choi_XZ',
                    basis='xz',
                    distance=12,
                    rounds=12,
                    error_rate=0.003,
                    gamma0=config['gamma0'],
                    pre_iter=config['pre_iter'],
                    num_sets=config['num_sets'],
                    set_max_iter=config['set_max_iter'],
                    gamma_dist_min=config['gamma_dist_interval'][0],
                    gamma_dist_max=config['gamma_dist_interval'][1],
                    stop_nconv=config['stop_nconv'],
                    target_errors=20,
                    batch=config.get('batch', 2048),
                    max_shots=100_000,
                    parallel=True,
                    backend=backend,
                    perf=perf,
                    dtype=dtype,
                )
                out['algo'] = 'relay'
                out['num_sets'] = config['num_sets']
                out['stop_nconv'] = config['stop_nconv']
                out['gamma0'] = config['gamma0']
                out['gamma_dist_interval'] = config['gamma_dist_interval']
                out['pre_iter'] = config['pre_iter']
                out['set_max_iter'] = config['set_max_iter']

            # Add configuration metadata
            out['config_name'] = config['name']
            out['backend'] = backend or ''
            out['perf'] = perf
            out['dtype'] = dtype
            
            # Print key results
            print(f"  LER: {out['logical_error_rate']:.2e}, Per-cycle LER: {out['per_cycle_logical_error_rate']:.2e}")
            if 'avg_legs' in out:
                print(f"  Avg BP iterations: {out['avg_bp_iterations']:.1f} (legs: {out['avg_legs']:.1f})")
            else:
                print(f"  Avg BP iterations: {out['avg_bp_iterations']:.1f}")
            return out
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def run_study(self):
        """Run the complete parameter sweep study for paper replication.
        
        This method executes the systematic parameter sweeps defined in define_configs(),
        running each configuration and saving results incrementally for robustness.
        """
        print("Starting Paper Study (plain + relay)...")
        print("=" * 50)
        
        # Build configuration list for systematic parameter sweeps
        configs: List[Dict[str, Any]] = self.define_configs()

        for i, config in enumerate(configs):
            tag = config.get('algo', '')
            print(f"\n[{i+1}/{len(configs)}] ({tag}) {config['name']}")
            result = self.run_any_config(config)
            if result is not None:
                self.results.append(result)
                self.save_result_incremental(result)

        # Save final consolidated results
        self.save_results()
    
    def save_results(self):
        """Save consolidated results to CSV and display summary table."""
        if not self.results:
            print("No results to save")
            return
        
        csv_path = self.output_dir / "paper_study_results.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self._csv_fieldnames())
            writer.writeheader()
            for result in self.results:
                writer.writerow(self._row_from_result(result))
        
        print(f"Results saved to: {csv_path}")
    
        # Display summary table matching paper format
        print("\nSummary Table:")
        print("=" * 100)
        print(f"{'R (num_sets)':<12} {'S (stop_nconv)':<15} {'BP Iterations':<15} {'LER':<12} {'Per-cycle LER':<15}")
        print("-" * 100)
        for result in self.results:
            r = result.get('num_sets', '-')
            s = result.get('stop_nconv', '-')
            avg_it = result.get('avg_bp_iterations', 0.0)
            ler = result.get('logical_error_rate', 0.0)
            pc_ler = result.get('per_cycle_logical_error_rate', 0.0)
            print(f"{str(r):<12} {str(s):<15} {avg_it:<15.1f} {ler:<12.2e} {pc_ler:<15.2e}")
    

def main():
    """Main function for paper study script."""
    study = RelayBPPaperStudy()
    study.run_study()


if __name__ == '__main__':
    main()