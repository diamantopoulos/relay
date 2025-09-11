#!/usr/bin/env python3
"""
Relay-BP Paper Study Script

This script replicates the paper's methodology exactly:
- X-axis: Average BP iteration count (not wall time)
- Y-axis: Per-cycle logical error rate (not per-shot)
- Parameters: S = solutions sought, R = max relay legs
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, List
import csv


class RelayBPPaperStudy:
    """Study class for replicating the paper's methodology."""
    
    def __init__(self, output_dir: str = "paper_study_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def define_parameter_gridOLD(self) -> List[Dict[str, Any]]:
        """Define the parameter grid matching the paper's methodology."""
        
        # Paper's Relay-BP variants: S = solutions sought (stop_nconv)
        paper_variants = [1, 3, 5, 7, 9]
        
        configs = []
        
        # Optional anchors removed; we now sweep R for each S
        
        # Sweep R values for each S (paper-style families)
        sweep_r_values = [50, 100, 200, 400, 600, 800]
        for s in paper_variants:
            for r in sweep_r_values:
                configs.append({
                    'name': f'Relay-BP-{s}-R{r}',
                    'num_sets': r,
                    'gamma0': 0.125,
                    'gamma_dist_interval': (-0.24, 0.66),
                    'pre_iter': 80,
                    'set_max_iter': 60,
                    'stop_nconv': s,
                })
        
        return configs
    
    def define_parameter_grid(self):
        # Sweep different R values (num_sets) while keeping S=1 constant
        num_sets_values = [1, 2, 3, 5]
        configs = []
        for r in num_sets_values:
            configs.append({
                'name': f'Relay-BP-R{r}',
                'num_sets': r,
                'gamma0': 0.125,
                'gamma_dist_interval': (-0.24, 0.66),
                'pre_iter': 80,
                'set_max_iter': 60,
                'stop_nconv': 1,  # Keep S=1 constant
            })
        return configs

    def run_single_config(self, config: Dict[str, Any], shots: int = 1000) -> Dict[str, Any]:
        """Run a single configuration using the detailed decoder."""
        
        print(f"Running {config['name']}...")
        
        # Build command using the detailed decoder
        cmd = [
            'python', 'relay_bp_detailed.py',
            '--circuit', 'bicycle_bivariate_144_12_12_memory_choi_XZ',
            '--basis', 'xz',
            '--error-rate', '0.003', '--distance', '12', '--rounds', '12',
            '--num-sets', str(config['num_sets']),
            '--gamma0', str(config['gamma0']),
            '--gamma-dist-min', str(config['gamma_dist_interval'][0]),
            '--gamma-dist-max', str(config['gamma_dist_interval'][1]),
            '--pre-iter', str(config['pre_iter']),
            '--set-max-iter', str(config['set_max_iter']),
            '--stop-nconv', str(config['stop_nconv']),
            '--target-errors', '20',
            '--batch', '2000',
            '--parallel',
            '--measure-time',
            '--output-format', 'json'
        ]
        
        try:
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse JSON output
            output_data = json.loads(result.stdout.strip())
            
            # Add configuration info
            output_data['config_name'] = config['name']
            output_data['num_sets'] = config['num_sets']
            output_data['stop_nconv'] = config['stop_nconv']
            output_data['gamma0'] = config['gamma0']
            output_data['gamma_dist_interval'] = config['gamma_dist_interval']
            
            print(f"  LER: {output_data['logical_error_rate']:.2e}, Per-cycle LER: {output_data['per_cycle_logical_error_rate']:.2e}")
            print(f"  Avg BP iterations: {output_data['avg_bp_iterations']:.1f} (legs: {output_data['avg_legs']:.1f})")
            
            return output_data
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def run_study(self, shots: int = 1000):
        """Run the complete study."""
        
        print("Starting Relay-BP Paper Study...")
        print("=" * 50)
        
        configs = self.define_parameter_grid()
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] {config['name']}")
            result = self.run_single_config(config, shots)
            if result is not None:
                self.results.append(result)
        
        print(f"\nStudy completed! Collected {len(self.results)} results.")
        
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
            fieldnames = ['config_name', 'num_sets', 'stop_nconv', 'gamma0', 'shots', 'logical_errors', 
                         'logical_error_rate', 'per_cycle_logical_error_rate', 'avg_bp_iterations', 'avg_legs']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow({k: result.get(k, '') for k in fieldnames})
        
        print(f"Results saved to: {csv_path}")
    
    def print_summary_table(self):
        """Print a summary table of results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\nSummary Table:")
        print("=" * 80)
        print(f"{'R (num_sets)':<12} {'BP Iterations':<15} {'Legs':<8} {'LER':<12} {'Per-cycle LER':<15}")
        print("-" * 80)
        for result in self.results:
            print(f"{result['num_sets']:<12} {result['avg_bp_iterations']:<15.1f} {result['avg_legs']:<8.1f} {result['logical_error_rate']:<12.2e} {result['per_cycle_logical_error_rate']:<15.2e}")
    
    def plot_performance_curves(self):
        """Print performance data for manual plotting."""
        
        if not self.results:
            print("No results to plot")
            return
        
        print("\nPerformance Data for Plotting:")
        print("=" * 60)
        print("R (num_sets) | BP Iterations | Legs | LER | Per-cycle LER")
        print("-" * 60)
        
        for result in self.results:
            print(f"{result['num_sets']:>11} | {result['avg_bp_iterations']:>13.1f} | {result['avg_legs']:>4.1f} | {result['logical_error_rate']:>3.2e} | {result['per_cycle_logical_error_rate']:>13.2e}")
        
        print("\nData saved to CSV for external plotting tools.")
        print("You can use the CSV file with tools like:")
        print("- Python matplotlib/pandas")
        print("- R")
        print("- Excel")
        print("- Any other plotting software")


def main():
    """Main function."""
    study = RelayBPPaperStudy()
    study.run_study(shots=50)  # Start with fewer shots for faster testing


if __name__ == '__main__':
    main()
