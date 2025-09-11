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
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


class RelayBPPaperStudy:
    """Study class for replicating the paper's methodology."""
    
    def __init__(self, output_dir: str = "paper_study_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        
    def define_parameter_grid(self) -> List[Dict[str, Any]]:
        """Define the parameter grid matching the paper's methodology."""
        
        # Paper's Relay-BP variants: S = solutions sought (stop_nconv)
        paper_variants = [1, 3, 5, 7, 9]
        
        configs = []
        
        # Optional anchors removed; we now sweep R for each S
        
        # Sweep R values for each S (paper-style families)
        sweep_r_values = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800]
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
    
    def run_single_config(self, config: Dict[str, Any], shots: int = 1000) -> Dict[str, Any]:
        """Run a single configuration using the detailed decoder."""
        
        print(f"Running {config['name']}...")
        
        # Build command using the detailed decoder
        cmd = [
            'python', 'relay_bp_detailed.py',
            '--num-sets', str(config['num_sets']),
            '--gamma0', str(config['gamma0']),
            '--gamma-dist-min', str(config['gamma_dist_interval'][0]),
            '--gamma-dist-max', str(config['gamma_dist_interval'][1]),
            '--pre-iter', str(config['pre_iter']),
            '--set-max-iter', str(config['set_max_iter']),
            '--stop-nconv', str(config['stop_nconv']),
            '--target-errors', '200',
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
        
        # Create plots
        self.plot_performance_curves()
    
    def save_results(self):
        """Save results to CSV."""
        if not self.results:
            print("No results to save")
            return
        
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "paper_study_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
    
    def plot_performance_curves(self):
        """Plot the performance curves matching the paper's methodology."""
        
        if not self.results:
            print("No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        plt.figure(figsize=(12, 8))
        
        # Define colors and markers
        colors = ['red', 'purple', 'brown', 'pink', 'grey', 'yellow', 'orange', 'green', 'blue']
        
        # Define variants for plotting (S values from paper)
        paper_variants = [1, 3, 5, 7, 9]  # S = solutions sought
        
        # Add reference lines from paper
        plt.axvline(x=600, color='grey', linestyle='--', alpha=0.7, linewidth=2, label='12µs @ 20ns/iter (≈600 iters)')
        plt.axhline(y=4e-4, color='grey', linestyle=':', alpha=0.7, linewidth=2, label='BP-10000/OSD-CS10 (asymptotic)')
        plt.axhline(y=3e-5, color='grey', linestyle=':', alpha=0.7, linewidth=2, label='Relay-BP-1000000-100 (asymptotic)')
        
        # Set labels and title
        plt.xlabel('Average BP iteration count', fontsize=14)
        plt.ylabel('Per-cycle logical error rate', fontsize=14)
        plt.title('Relay-BP Decoder Performance (Paper Methodology) at Physical Error Rate p = 3 × 10⁻³', fontsize=16)
        
        # Plot paper variants (S = 1, 3, 5, 7, 9)
        for i, s in enumerate(paper_variants):
            variant_data = df[df['stop_nconv'] == s]
            if not variant_data.empty:
                print(f"Plotting Relay-BP-{s}: {len(variant_data)} points")
                print(f"  BP iteration range: {variant_data['avg_bp_iterations'].min():.1f} - {variant_data['avg_bp_iterations'].max():.1f}")
                print(f"  Per-cycle LER range: {variant_data['per_cycle_logical_error_rate'].min():.2e} - {variant_data['per_cycle_logical_error_rate'].max():.2e}")

                # Handle zero logical error rates for log scale
                ler_values = variant_data['per_cycle_logical_error_rate'].copy()
                ler_values = ler_values.clip(lower=1e-6)  # Replace 0 with small positive value

                plt.loglog(
                    variant_data['avg_bp_iterations'], # Using BP iteration count (x-axis)
                    ler_values,  # Using per-cycle LER (y-axis)
                    marker='+',
                    color=colors[i % len(colors)],
                    label=f'Relay-BP-{s}',
                    markersize=10,
                    linewidth=2,
                    linestyle='-'
                )
        
        # Plot additional R values (at fixed S=1) with different style
        additional_r_data = df[df['stop_nconv'] == 1]
        if not additional_r_data.empty:
            # Handle zero logical error rates for log scale
            ler_values = additional_r_data['per_cycle_logical_error_rate'].copy()
            ler_values = ler_values.clip(lower=1e-6)  # Replace 0 with small positive value

            plt.loglog(
                additional_r_data['avg_bp_iterations'],
                ler_values,
                marker='o',
                color='blue',
                label=f'Relay-BP (R sweep, S=1)',
                markersize=6,
                linewidth=1,
                linestyle='--',
                alpha=0.7
            )
        
        # Set axis limits based on actual data
        if not df.empty:
            x_min = df['avg_bp_iterations'].min() * 0.5
            x_max = df['avg_bp_iterations'].max() * 2

            # Handle zero logical error rates for log scale
            min_ler = df['per_cycle_logical_error_rate'].min()
            max_ler = df['per_cycle_logical_error_rate'].max()

            # If min_ler is 0, set it to a small positive value
            if min_ler == 0:
                min_ler = 1e-6  # Small positive value for log scale

            y_min = min_ler * 0.1
            y_max = max_ler * 10 if max_ler > 0 else 1e-2
        else:
            x_min, x_max = 1, 1000
            y_min, y_max = 1e-6, 1

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Add legend and grid
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / "relay_bp_paper_performance_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to {plot_path}")
        
        plt.show()


def main():
    """Main function."""
    study = RelayBPPaperStudy()
    study.run_study(shots=100)  # Start with fewer shots for testing


if __name__ == '__main__':
    main()
