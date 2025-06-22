#!/usr/bin/env python3
"""
MPI-based Distributed Matrix Multiplication
Implements row-wise partitioning strategy
"""

from mpi4py import MPI
import numpy as np
import time
import argparse
import json

def initialize_matrices(size, rank):
    """Initialize matrices A and B on root process"""
    if rank == 0:
        np.random.seed(42)
        A = np.random.rand(size, size).astype(np.float64)
        B = np.random.rand(size, size).astype(np.float64)
        return A, B
    else:
        return None, None

def distribute_matrix_rows(A, size, comm, rank, num_procs):
    """
    Distribute rows of matrix A among processes
    Uses row-wise partitioning strategy
    """
    rows_per_proc = size // num_procs
    remainder = size % num_procs
    
    # Calculate number of rows for each process
    if rank < remainder:
        local_rows = rows_per_proc + 1
        start_row = rank * (rows_per_proc + 1)
    else:
        local_rows = rows_per_proc
        start_row = remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc
    
    # Prepare local matrix A
    local_A = np.zeros((local_rows, size), dtype=np.float64)
    
    if rank == 0:
        # Send rows to other processes
        for proc in range(1, num_procs):
            if proc < remainder:
                proc_rows = rows_per_proc + 1
                proc_start = proc * (rows_per_proc + 1)
            else:
                proc_rows = rows_per_proc
                proc_start = remainder * (rows_per_proc + 1) + (proc - remainder) * rows_per_proc
            
            if proc_rows > 0:
                comm.send(A[proc_start:proc_start + proc_rows], dest=proc, tag=proc)
        
        # Keep local rows
        local_A = A[start_row:start_row + local_rows].copy()
    else:
        # Receive rows from root
        if local_rows > 0:
            local_A = comm.recv(source=0, tag=rank)
    
    return local_A, local_rows, start_row

def mpi_matrix_multiply(size, comm, rank, num_procs):
    """
    Distributed matrix multiplication using MPI
    """
    # Initialize matrices on root
    A, B = initialize_matrices(size, rank)
    
    # Broadcast matrix B to all processes
    B = comm.bcast(B, root=0)
    
    # Distribute rows of matrix A
    local_A, local_rows, start_row = distribute_matrix_rows(A, size, comm, rank, num_procs)
    
    # Perform local matrix multiplication
    local_C = np.zeros((local_rows, size), dtype=np.float64)
    if local_rows > 0:
        local_C = np.dot(local_A, B)
    
    # Gather results at root
    if rank == 0:
        C = np.zeros((size, size), dtype=np.float64)
        # Copy local result
        C[start_row:start_row + local_rows] = local_C
        
        # Receive results from other processes
        for proc in range(1, num_procs):
            if proc < size % num_procs:
                proc_rows = size // num_procs + 1
                proc_start = proc * (size // num_procs + 1)
            else:
                proc_rows = size // num_procs
                proc_start = (size % num_procs) * (size // num_procs + 1) + \
                           (proc - size % num_procs) * (size // num_procs)
            
            if proc_rows > 0:
                received_data = comm.recv(source=proc, tag=proc + 100)
                C[proc_start:proc_start + proc_rows] = received_data
        
        return C
    else:
        # Send local result to root
        if local_rows > 0:
            comm.send(local_C, dest=0, tag=rank + 100)
        return None

def benchmark_mpi(matrix_sizes, num_runs, comm, rank, num_procs):
    """Benchmark MPI matrix multiplication"""
    results = {}
    
    for size in matrix_sizes:
        if rank == 0:
            print(f"Benchmarking MPI multiplication for {size}x{size} matrices with {num_procs} processes...")
        
        times = []
        for run in range(num_runs):
            comm.Barrier()  # Synchronize all processes
            start_time = time.time()
            
            C = mpi_matrix_multiply(size, comm, rank, num_procs)
            
            comm.Barrier()  # Synchronize all processes
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append(execution_time)
            
            if rank == 0:
                print(f"  Run {run+1}: {execution_time:.4f} seconds")
        
        if rank == 0:
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[size] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'times': times,
                'num_processes': num_procs
            }
            print(f"  Average: {avg_time:.4f} ± {std_time:.4f} seconds\n")
    
    return results

def verify_correctness(size, comm, rank):
    """Verify correctness of MPI implementation against serial implementation"""
    if rank == 0:
        print(f"Verifying correctness for {size}x{size} matrices...")
        
        # Generate test matrices
        np.random.seed(42)
        A = np.random.rand(size, size).astype(np.float64)
        B = np.random.rand(size, size).astype(np.float64)
        
        # Serial computation
        C_serial = np.dot(A, B)
    
    # MPI computation
    C_mpi = mpi_matrix_multiply(size, comm, rank, comm.Get_size())
    
    if rank == 0:
        # Compare results
        if np.allclose(C_serial, C_mpi, rtol=1e-10, atol=1e-10):
            print("MPI implementation is correct!")
            return True
        else:
            print("MPI implementation has errors!")
            max_diff = np.max(np.abs(C_serial - C_mpi))
            print(f"  Maximum difference: {max_diff}")
            return False
    
    return None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    
    parser = argparse.ArgumentParser(description='MPI Matrix Multiplication Benchmark')
    parser.add_argument('--sizes', nargs='+', type=int, default=[100, 200, 400],
                       help='Matrix sizes to benchmark')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs for each size')
    parser.add_argument('--verify', action='store_true',
                       help='Verify correctness against serial implementation')
    parser.add_argument('--output', type=str, default='mpi_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    if rank == 0:
        print("MPI Matrix Multiplication Benchmark")
        print("=" * 40)
        print(f"Running with {num_procs} processes")
        print()
    
    # Verify correctness if requested
    if args.verify:
        verify_correctness(min(args.sizes), comm, rank)
        if rank == 0:
            print()
    
    # Run benchmarks
    results = benchmark_mpi(args.sizes, args.runs, comm, rank, num_procs)
    
    # Save results (only from root process)
    if rank == 0 and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
        
        print("\nBenchmark Summary:")
        print("-" * 30)
        for size, data in results.items():
            print(f"{size}x{size}: {data['avg_time']:.4f} ± {data['std_time']:.4f} seconds")

if __name__ == "__main__":
    main()