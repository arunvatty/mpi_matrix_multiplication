#!/usr/bin/env python3
"""
Serial Matrix Multiplication Implementation
For benchmarking against the distributed MPI version
"""

import numpy as np
import time
import argparse
import json

def serial_matrix_multiply(A, B):
    """
    Standard serial matrix multiplication
    Args:
        A: First matrix (m x k)
        B: Second matrix (k x n)
    Returns:
        C: Result matrix (m x n)
    """
    return np.dot(A, B)

def generate_random_matrix(rows, cols, seed=42):
    """Generate a random matrix with given dimensions"""
    np.random.seed(seed)
    return np.random.rand(rows, cols)

def benchmark_serial(matrix_sizes, num_runs=3):
    """
    Benchmark serial matrix multiplication for different matrix sizes
    """
    results = {}
    
    for size in matrix_sizes:
        print(f"Benchmarking serial multiplication for {size}x{size} matrices...")
        times = []
        
        for run in range(num_runs):
            # Generate matrices
            A = generate_random_matrix(size, size, seed=42+run)
            B = generate_random_matrix(size, size, seed=100+run)
            
            # Time the multiplication
            start_time = time.time()
            C = serial_matrix_multiply(A, B)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            print(f"  Run {run+1}: {execution_time:.4f} seconds")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        results[size] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'times': times
        }
        print(f"  Average: {avg_time:.4f} ± {std_time:.4f} seconds\n")
    
    return results

def save_results(results, filename):
    """Save benchmarking results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Serial Matrix Multiplication Benchmark')
    parser.add_argument('--sizes', nargs='+', type=int, default=[100, 200, 400, 800],
                       help='Matrix sizes to benchmark')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs for each size')
    parser.add_argument('--output', type=str, default='serial_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("Serial Matrix Multiplication Benchmark")
    print("=" * 40)
    
    results = benchmark_serial(args.sizes, args.runs)
    save_results(results, args.output)
    
    print("\nBenchmark Summary:")
    print("-" * 30)
    for size, data in results.items():
        print(f"{size}x{size}: {data['avg_time']:.4f} ± {data['std_time']:.4f} seconds")

if __name__ == "__main__":
    main()