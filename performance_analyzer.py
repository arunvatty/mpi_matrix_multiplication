#!/usr/bin/env python3
"""
Performance Analysis and Visualization Tool
Compares serial vs MPI performance and generates reports
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

class PerformanceAnalyzer:
    """Analyzes and visualizes performance data from benchmarks"""
    
    def __init__(self):
        self.serial_data = {}
        self.mpi_data = {}
        self.results = {}
    
    def load_data(self, serial_file, mpi_files):
        """Load performance data from JSON files"""
        # Load serial data
        if Path(serial_file).exists():
            with open(serial_file, 'r') as f:
                self.serial_data = json.load(f)
            print(f"Loaded serial data from {serial_file}")
        else:
            print(f"Warning: Serial data file {serial_file} not found")
        
        # Load MPI data (can be multiple files for different process counts)
        self.mpi_data = {}
        for mpi_file in mpi_files:
            if Path(mpi_file).exists():
                with open(mpi_file, 'r') as f:
                    data = json.load(f)
                    # Extract number of processes from the data
                    num_procs = list(data.values())[0].get('num_processes', 'unknown')
                    self.mpi_data[num_procs] = data
                print(f"Loaded MPI data from {mpi_file} ({num_procs} processes)")
            else:
                print(f"Warning: MPI data file {mpi_file} not found")
    
    def calculate_speedup(self):
        """Calculate speedup and efficiency metrics"""
        self.results = {}
        
        for matrix_size in self.serial_data.keys():
            matrix_size_int = int(matrix_size)
            self.results[matrix_size_int] = {
                'serial_time': self.serial_data[matrix_size]['avg_time'],
                'mpi_results': {}
            }
            
            for num_procs, mpi_data in self.mpi_data.items():
                if matrix_size in mpi_data:
                    mpi_time = mpi_data[matrix_size]['avg_time']
                    speedup = self.serial_data[matrix_size]['avg_time'] / mpi_time
                    efficiency = speedup / num_procs
                    
                    self.results[matrix_size_int]['mpi_results'][num_procs] = {
                        'time': mpi_time,
                        'speedup': speedup,
                        'efficiency': efficiency
                    }
    
    def generate_performance_plots(self, output_dir='plots'):
        """Generate performance visualization plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot 1: Execution Time Comparison
        self.plot_execution_times(output_dir)
        
        # Plot 2: Speedup Analysis
        self.plot_speedup(output_dir)
        
        # Plot 3: Efficiency Analysis
        self.plot_efficiency(output_dir)
        
        # Plot 4: Scalability Analysis
        self.plot_scalability(output_dir)
    
    def plot_execution_times(self, output_dir):
        """Plot execution times for different matrix sizes"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        matrix_sizes = sorted(self.results.keys())
        
        # Plot serial times
        serial_times = [self.results[size]['serial_time'] for size in matrix_sizes]
        ax.plot(matrix_sizes, serial_times, 'o-', label='Serial', linewidth=2, markersize=8)
        
        # Plot MPI times for different process counts
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, num_procs in enumerate(sorted(self.mpi_data.keys())):
            mpi_times = []
            for size in matrix_sizes:
                if num_procs in self.results[size]['mpi_results']:
                    mpi_times.append(self.results[size]['mpi_results'][num_procs]['time'])
                else:
                    mpi_times.append(None)
            
            # Filter out None values
            valid_sizes = [size for size, time in zip(matrix_sizes, mpi_times) if time is not None]
            valid_times = [time for time in mpi_times if time is not None]
            
            if valid_times:
                color = colors[i % len(colors)]
                ax.plot(valid_sizes, valid_times, 'o-', label=f'MPI ({num_procs} processes)', 
                       linewidth=2, markersize=8, color=color)
        
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_title('Matrix Multiplication Performance Comparison', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/execution_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_speedup(self, output_dir):
        """Plot speedup analysis"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        matrix_sizes = sorted(self.results.keys())
        
        # Plot speedup for different process counts
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, num_procs in enumerate(sorted(self.mpi_data.keys())):
            speedups = []
            for size in matrix_sizes:
                if num_procs in self.results[size]['mpi_results']:
                    speedups.append(self.results[size]['mpi_results'][num_procs]['speedup'])
                else:
                    speedups.append(None)
            
            # Filter out None values
            valid_sizes = [size for size, speedup in zip(matrix_sizes, speedups) if speedup is not None]
            valid_speedups = [speedup for speedup in speedups if speedup is not None]
            
            if valid_speedups:
                color = colors[i % len(colors)]
                ax.plot(valid_sizes, valid_speedups, 'o-', label=f'{num_procs} processes', 
                       linewidth=2, markersize=8, color=color)
        
        # Plot ideal speedup line
        if self.mpi_data:
            max_procs = max(self.mpi_data.keys())
            ax.axhline(y=max_procs, color='black', linestyle='--', alpha=0.5, 
                      label=f'Ideal speedup ({max_procs}x)')
        
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Speedup Analysis', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_efficiency(self, output_dir):
        """Plot efficiency analysis"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        matrix_sizes = sorted(self.results.keys())
        
        # Plot efficiency for different process counts
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i, num_procs in enumerate(sorted(self.mpi_data.keys())):
            efficiencies = []
            for size in matrix_sizes:
                if num_procs in self.results[size]['mpi_results']:
                    efficiencies.append(self.results[size]['mpi_results'][num_procs]['efficiency'])
                else:
                    efficiencies.append(None)
            
            # Filter out None values
            valid_sizes = [size for size, eff in zip(matrix_sizes, efficiencies) if eff is not None]
            valid_efficiencies = [eff for eff in efficiencies if eff is not None]
            
            if valid_efficiencies:
                color = colors[i % len(colors)]
                ax.plot(valid_sizes, valid_efficiencies, 'o-', label=f'{num_procs} processes', 
                       linewidth=2, markersize=8, color=color)
        
        # Plot ideal efficiency line (100%)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal efficiency (100%)')
        
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Efficiency', fontsize=12)
        ax.set_title('Parallel Efficiency Analysis', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_scalability(self, output_dir):
        """Plot scalability analysis"""
        if not self.mpi_data or len(self.mpi_data) < 2:
            print("Insufficient data for scalability analysis")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Choose a representative matrix size (largest available)
        matrix_sizes = sorted(self.results.keys())
        target_size = matrix_sizes[-1]
        
        process_counts = sorted(self.mpi_data.keys())
        speedups = []
        efficiencies = []
        
        for num_procs in process_counts:
            if num_procs in self.results[target_size]['mpi_results']:
                speedups.append(self.results[target_size]['mpi_results'][num_procs]['speedup'])
                efficiencies.append(self.results[target_size]['mpi_results'][num_procs]['efficiency'])
            else:
                speedups.append(None)
                efficiencies.append(None)
        
        # Filter out None values
        valid_procs = [p for p, s in zip(process_counts, speedups) if s is not None]
        valid_speedups = [s for s in speedups if s is not None]
        valid_efficiencies = [e for e in efficiencies if e is not None]
        
        if valid_speedups:
            ax.plot(valid_procs, valid_speedups, 'o-', label='Actual Speedup', 
                   linewidth=2, markersize=8, color='blue')
            ax.plot(valid_procs, valid_procs, '--', label='Ideal Speedup', 
                   linewidth=2, color='red', alpha=0.7)
        
        ax.set_xlabel('Number of Processes', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title(f'Scalability Analysis (Matrix Size: {target_size}x{target_size})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file='performance_report.txt'):
        """Generate a detailed performance report"""
        with open(output_file, 'w') as f:
            f.write("MPI Matrix Multiplication Performance Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            f.write("Performance Summary:\n")
            f.write("-" * 20 + "\n")
            
            for matrix_size in sorted(self.results.keys()):
                f.write(f"\nMatrix Size: {matrix_size}x{matrix_size}\n")
                f.write(f"Serial Time: {self.results[matrix_size]['serial_time']:.4f} seconds\n")
                
                for num_procs in sorted(self.results[matrix_size]['mpi_results'].keys()):
                    mpi_result = self.results[matrix_size]['mpi_results'][num_procs]
                    f.write(f"MPI ({num_procs} processes):\n")
                    f.write(f"  Time: {mpi_result['time']:.4f} seconds\n")
                    f.write(f"  Speedup: {mpi_result['speedup']:.2f}x\n")
                    f.write(f"  Efficiency: {mpi_result['efficiency']:.2f} ({mpi_result['efficiency']*100:.1f}%)\n")
            
            # Best performance analysis
            f.write("\n\nBest Performance Analysis:\n")
            f.write("-" * 30 + "\n")
            
            best_speedups = {}
            for matrix_size in sorted(self.results.keys()):
                best_speedup = 0
                best_procs = 0
                for num_procs, result in self.results[matrix_size]['mpi_results'].items():
                    if result['speedup'] > best_speedup:
                        best_speedup = result['speedup']
                        best_procs = num_procs
                
                if best_speedup > 0:
                    best_speedups[matrix_size] = (best_procs, best_speedup)
                    f.write(f"Matrix {matrix_size}x{matrix_size}: Best speedup {best_speedup:.2f}x with {best_procs} processes\n")
            
            # Recommendations
            f.write("\n\nRecommendations:\n")
            f.write("-" * 15 + "\n")
            
            if best_speedups:
                avg_best_speedup = np.mean([speedup for _, speedup in best_speedups.values()])
                f.write(f"Average best speedup: {avg_best_speedup:.2f}x\n")
                
                if avg_best_speedup > 2.0:
                    f.write("Good parallel performance achieved\n")
                elif avg_best_speedup > 1.5:
                    f.write("Moderate parallel performance - consider optimizing communication\n")
                else:
                    f.write("Poor parallel performance - review algorithm and communication strategy\n")
        
        print(f"Performance report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Performance Analysis Tool')
    parser.add_argument('--serial', type=str, default='serial_results.json',
                       help='Serial benchmark results file')
    parser.add_argument('--mpi', nargs='+', default=['mpi_results.json'],
                       help='MPI benchmark results files')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--report', type=str, default='performance_report.txt',
                       help='Output file for performance report')
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer()
    analyzer.load_data(args.serial, args.mpi)
    analyzer.calculate_speedup()
    analyzer.generate_performance_plots(args.output_dir)
    analyzer.generate_report(args.report)
    
    print(f"Analysis complete! Check {args.output_dir}/ for plots and {args.report} for detailed report.")

if __name__ == "__main__":
    main()