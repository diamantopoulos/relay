#!/usr/bin/env python3
"""
Relay-BP Paper Study Script

This script replicates the paper's methodology exactly:
- X-axis: Average BP iteration count (not wall time)
- Y-axis: Per-cycle logical error rate (not per-shot)
- Parameters: S = solutions sought, R = max relay legs
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import csv

# Import the reusable function from relay_bp_detailed
from relay_bp_detailed import run_relay_bp_experiment, run_plain_bp_experiment


class RelayBPPaperStudy:
    """Study class for replicating the paper's methodology."""
    
    def __init__(self, output_dir: str = "paper_study_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    
    def define_parameter_grid(self):
        # Fix S (stop_nconv) and sweep R (num_sets) broadly per the paper
        stop_nconv_values = [1, 2, 3, 5, 7, 9]
        # Use an odd sweep for R to populate x-axis (adjust upper bound as needed)
        num_sets_values = list(range(1, 61, 2))  # e.g., 1,3,5,...,59
        configs = []
        for s in stop_nconv_values:
            for r in num_sets_values:
                configs.append({
                    'name': f'Relay-BP-S{s}-R{r}',
                    'num_sets': r,
                    'gamma0': 0.125,
                    'gamma_dist_interval': (-0.24, 0.66),
                    'pre_iter': 80,
                    'set_max_iter': 60,
                    'stop_nconv': s,
                })
        return configs

    def define_plain_bp_grid(self):
        # Sweep plain BP (no memory, no relay) by max_iter to populate x-axis
        max_iter_values = list(range(20, 361, 40))  # 20,40,...,360
        configs = []
        for tmax in max_iter_values:
            configs.append({
                'name': f'PlainBP-maxiter{tmax}',
                'max_iter': tmax,
                'alpha': None,
            })
        return configs

    def run_single_config(self, config: Dict[str, Any], shots: int = 1000) -> Dict[str, Any]:
        """Run a single configuration using the detailed decoder."""
        
        print(f"Running {config['name']}...")
        
        try:
            # Run the Relay-BP experiment directly using the reusable function
            output_data = run_relay_bp_experiment(
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
                target_errors=100,
                batch=20000,
                max_shots=1000000,
                parallel=True,
                measure_time=True
            )
            
            # Add configuration info
            output_data['config_name'] = config['name']
            output_data['num_sets'] = config['num_sets']
            output_data['stop_nconv'] = config['stop_nconv']
            output_data['gamma0'] = config['gamma0']
            output_data['gamma_dist_interval'] = config['gamma_dist_interval']
            output_data['pre_iter'] = config['pre_iter']
            output_data['set_max_iter'] = config['set_max_iter']
            
            
            print(f"  LER: {output_data['logical_error_rate']:.2e}, Per-cycle LER: {output_data['per_cycle_logical_error_rate']:.2e}")
            print(f"  Avg BP iterations: {output_data['avg_bp_iterations']:.1f} (legs: {output_data['avg_legs']:.1f})")
            
            return output_data
            
        except Exception as e:
            print(f"  Error: {e}")
            return None

    def run_single_plain_bp_config(self, config: Dict[str, Any], shots: int = 1000) -> Dict[str, Any]:
        print(f"Running {config['name']}...")
        try:
            output_data = run_plain_bp_experiment(
                circuit='bicycle_bivariate_144_12_12_memory_choi_XZ',
                basis='xz',
                distance=12,
                rounds=12,
                error_rate=0.003,
                max_iter=config['max_iter'],
                alpha=config.get('alpha', None),
                target_errors=200,
                batch=10_000,
                max_shots=2_000_000,
                parallel=True,
                measure_time=True
            )
            output_data['config_name'] = config['name']
            output_data['max_iter'] = config['max_iter']
            print(f"  LER: {output_data['logical_error_rate']:.2e}, Per-cycle LER: {output_data['per_cycle_logical_error_rate']:.2e}")
            print(f"  Avg BP iterations: {output_data['avg_bp_iterations']:.1f}")
            return output_data
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def run_study(self, shots: int = 1000):
        """Run the complete study."""
        
        print("Starting Relay-BP Paper Study...")
        print("=" * 50)
        
        configs = self.define_parameter_grid()
        
        #for i, config in enumerate(configs):
        #    print(f"\n[{i+1}/{len(configs)}] {config['name']}")
       #     result = self.run_single_config(config, shots)
       #     if result is not None:
       #         self.results.append(result)
        
        #print(f"\nStudy completed! Collected {len(self.results)} results.")

        # Run plain BP sweep (paper baseline)
        bp_configs = self.define_plain_bp_grid()
        for i, config in enumerate(bp_configs):
            print(f"\n[Plain BP {i+1}/{len(bp_configs)}] {config['name']}")
            result = self.run_single_plain_bp_config(config, shots)
            if result is not None:
                self.results.append(result)
        
        print(f"\nPlain BP Study completed! Collected {len(self.results)} results.")

        # Save results
        self.save_results()
        
        # Print summary table
        self.print_summary_table()
        
        # Create plots
        self.plot_performance_curves()
    
    def save_results(self):
        """Save results to CSV."""
        if not self.results:
            print("No results to save")
            return
        
        csv_path = self.output_dir / "paper_study_results.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                # Configuration parameters
                'config_name', 'num_sets', 'stop_nconv', 'gamma0', 'gamma_dist_min', 'gamma_dist_max',
                'pre_iter', 'set_max_iter', 'circuit', 'distance', 'rounds', 'error_rate', 'basis',
                'max_iter',
                # Experiment results
                'shots', 'logical_errors', 'logical_error_rate', 'per_cycle_logical_error_rate',
                'avg_bp_iterations', 'runtime_per_shot_ns'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                # Extract all parameters from config and results
                row = {
                    'config_name': result.get('config_name', ''),
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
                    'basis': 'xz',  # Fixed for this study
                    'max_iter': result.get('config', {}).get('max_iter', ''),
                    'shots': result.get('shots', ''),
                    'logical_errors': result.get('logical_errors', ''),
                    'logical_error_rate': result.get('logical_error_rate', ''),
                    'per_cycle_logical_error_rate': result.get('per_cycle_logical_error_rate', ''),
                    'avg_bp_iterations': result.get('avg_bp_iterations', ''),
                    'runtime_per_shot_ns': result.get('runtime_per_shot_ns', '')
                }
                writer.writerow(row)
        
        print(f"Results saved to: {csv_path}")
    
    def print_summary_table(self):
        """Print a summary table of results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\nSummary Table:")
        print("=" * 100)
        print(f"{'R (num_sets)':<12} {'S (stop_nconv)':<15} {'BP Iterations':<15} {'LER':<12} {'Per-cycle LER':<15}")
        print("-" * 100)
        for result in self.results:
            print(f"{result['num_sets']:<12} {result['stop_nconv']:<15} {result['avg_bp_iterations']:<15.1f} {result['logical_error_rate']:<12.2e} {result['per_cycle_logical_error_rate']:<15.2e}")
    
    def plot_performance_curves(self):
        """Print performance data for manual plotting."""
        
        if not self.results:
            print("No results to plot")
            return


def main():
    """Main function."""
    study = RelayBPPaperStudy()
    study.run_study(shots=50)  # Start with fewer shots for faster testing


if __name__ == '__main__':
    main()