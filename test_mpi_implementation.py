#!/usr/bin/env python3
"""
Test Suite for MPI Matrix Multiplication Implementation
Comprehensive testing including correctness, performance, and edge cases
"""

from mpi4py import MPI
import numpy as np
import time
import sys
import json
from pathlib import Path

# Import our modules
from mpi_matrix_multiplication import mpi_matrix_multiply, initialize_matrices
from serial_matrix_multiplication import serial_matrix_multiply

class MPITestSuite:
    """Comprehensive test suite for MPI matrix multiplication"""
    
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        self.tests_passed = 0
        self.tests_failed = 0
    
    def log(self, message, rank_filter=0):
        """Log message from specific rank only"""
        if self.rank == rank_filter:
            print(message)
    
    def test_correctness(self, sizes=[10, 50, 100]):
        """Test correctness against serial implementation"""
        self.log("Testing correctness against serial implementation...")
        
        for size in sizes:
            self.log(f"  Testing {size}x{size} matrices...")
            
            # Generate test matrices on root
            if self.rank == 0:
                np.random.seed(42)  # Fixed seed for reproducibility
                A = np.random.rand(size, size).astype(np.float64)
                B = np.random.rand(size, size).astype(np.float64)
                
                # Compute serial result
                C_serial = serial_matrix_multiply(A, B)
            else:
                A = B = C_serial = None
            
            # Compute MPI result
            C_mpi = mpi_matrix_multiply(size, self.comm, self.rank, self.num_procs)
            
            # Compare results on root
            if self.rank == 0:
                if np.allclose(C_serial, C_mpi, rtol=1e-10, atol=1e-10):
                    self.log(f"    ✓ PASSED: {size}x{size} matrices")
                    self.tests_passed += 1
                else:
                    max_diff = np.max(np.abs(C_serial - C_mpi))
                    self.log(f"    ✗ FAILED: {size}x{size} matrices (max diff: {max_diff})")
                    self.tests_failed += 1
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        self.log("Testing edge cases...")
        
        # Test with single process (should work like serial)
        if self.num_procs == 1:
            self.log("  Testing single process execution...")
            try:
                C = mpi_matrix_multiply(50, self.comm, self.rank, self.num_procs)
                if self.rank == 0 and C is not None:
                    self.log("    ✓ PASSED: Single process execution")
                    self.tests_passed += 1
                else:
                    self.log("    ✗ FAILED: Single process execution")
                    self.tests_failed += 1
            except Exception as e:
                self.log(f"    ✗ FAILED: Single process execution - {e}")
                self.tests_failed += 1
        
        # Test with matrix size smaller than number of processes
        small_size = max(2, self.num_procs - 1)
        self.log(f"  Testing small matrix ({small_size}x{small_size}) with {self.num_procs} processes...")
        try:
            C = mpi_matrix_multiply(small_size, self.comm, self.rank, self.num_procs)
            
            if self.rank == 0:
                if C is not None and C.shape == (small_size, small_size):
                    self.log("    ✓ PASSED: Small matrix test")
                    self.tests_passed += 1
                else:
                    self.log("    ✗ FAILED: Small matrix test")
                    self.tests_failed += 1
        except Exception as e:
            self.log(f"    ✗ FAILED: Small matrix test - {e}")
            self.tests_failed += 1
    
    def test_data_types(self):
        """Test different data types and precision"""
        self.log("Testing data type handling...")
        
        size = 20
        
        # Test with different numpy dtypes
        dtypes = [np.float32, np.float64]
        
        for dtype in dtypes:
            self.log(f"  Testing with {dtype.__name__}...")
            
            if self.rank == 0:
                np.random.seed(42)
                A = np.random.rand(size, size).astype(dtype)
                B = np.random.rand(size, size).astype(dtype)
                C_serial = np.dot(A, B)
            
            try:
                C_mpi = mpi_matrix_multiply(size, self.comm, self.rank, self.num_procs)
                
                if self.rank == 0:
                    if np.allclose(C_serial, C_mpi, rtol=1e-5, atol=1e-5):
                        self.log(f"    ✓ PASSED: {dtype.__name__} precision")
                        self.tests_passed += 1
                    else:
                        self.log(f"    ✗ FAILED: {dtype.__name__} precision")
                        self.tests_failed += 1
            except Exception as e:
                self.log(f"    ✗ FAILED: {dtype.__name__} precision - {e}")
                self.tests_failed += 1
    
    def test_performance_consistency(self):
        """Test performance consistency across multiple runs"""
        self.log("Testing performance consistency...")
        
        size = 100
        num_runs = 3
        times = []
        
        for run in range(num_runs):
            self.comm.Barrier()
            start_time = time.time()
            
            C = mpi_matrix_multiply(size, self.comm, self.rank, self.num_procs)
            
            self.comm.Barrier()
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        if self.rank == 0:
            avg_time = np.mean(times)
            std_time = np.std(times)
            cv = std_time / avg_time  # Coefficient of variation
            
            self.log(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
            self.log(f"  Coefficient of variation: {cv:.3f}")
            
            if cv < 0.1:  # Less than 10% variation
                self.log("    ✓ PASSED: Performance consistency")
                self.tests_passed += 1
            else:
                self.log("    ✗ FAILED: High performance variation")
                self.tests_failed += 1
    
    def test_communication_patterns(self):
        """Test MPI communication patterns"""
        self.log("Testing MPI communication patterns...")
        
        # Test broadcast functionality
        test_data = None
        if self.rank == 0:
            test_data = np.random.rand(10, 10)
        
        try:
            broadcasted_data = self.comm.bcast(test_data, root=0)
            
            if broadcasted_data is not None and broadcasted_data.shape == (10, 10):
                self.log("    ✓ PASSED: Broadcast communication")
                self.tests_passed += 1
            else:
                self.log("    ✗ FAILED: Broadcast communication")
                self.tests_failed += 1
        except Exception as e:
            self.log(f"    ✗ FAILED: Broadcast communication - {e}")
            self.tests_failed += 1
        
        # Test point-to-point communication
        try:
            if self.rank == 0 and self.num_procs > 1:
                test_array = np.array([1, 2, 3, 4, 5])
                self.comm.send(test_array, dest=1, tag=99)
                received = self.comm.recv(source=1, tag=100)
                
                if np.array_equal(received, test_array * 2):
                    self.log("    ✓ PASSED: Point-to-point communication")
                    self.tests_passed += 1
                else:
                    self.log("    ✗ FAILED: Point-to-point communication")
                    self.tests_failed += 1
            elif self.rank == 1:
                received = self.comm.recv(source=0, tag=99)
                self.comm.send(received * 2, dest=0, tag=100)
        except Exception as e:
            if self.rank == 0:
                self.log(f"    ✗ FAILED: Point-to-point communication - {e}")
                self.tests_failed += 1
    
    def benchmark_scalability(self):
        """Quick scalability benchmark"""
        self.log("Running scalability benchmark...")
        
        sizes = [50, 100, 200]
        results = {}
        
        for size in sizes:
            self.comm.Barrier()
            start_time = time.time()
            
            C = mpi_matrix_multiply(size, self.comm, self.rank, self.num_procs)
            
            self.comm.Barrier()
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[size] = execution_time
            
            if self.rank == 0:
                self.log(f"  {size}x{size}: {execution_time:.4f} seconds")
        
        # Save results
        if self.rank == 0:
            with open(f'test_results_{self.num_procs}p.json', 'w') as f:
                json.dump({
                    'num_processes': self.num_procs,
                    'benchmark_results': results,
                    'tests_passed': self.tests_passed,
                    'tests_failed': self.tests_failed
                }, f, indent=2)
    
    def run_all_tests(self):
        """Run all tests in the suite"""
        self.log("=" * 50)
        self.log(f"MPI Matrix Multiplication Test Suite")
        self.log(f"Running on {self.num_procs} processes")
        self.log("=" * 50)
        
        try:
            self.test_correctness()
            self.test_edge_cases()
            self.test_data_types()
            self.test_performance_consistency()
            if self.num_procs > 1:
                self.test_communication_patterns()
            self.benchmark_scalability()
            
        except Exception as e:
            self.log(f"Test suite error: {e}")
            self.tests_failed += 1
        
        # Summary
        self.log("\n" + "=" * 50)
        self.log("Test Summary:")
        self.log(f"Tests Passed: {self.tests_passed}")
        self.log(f"Tests Failed: {self.tests_failed}")
        self.log(f"Success Rate: {self.tests_passed/(self.tests_passed + self.tests_failed)*100:.1f}%")
        
        if self.tests_failed == 0:
            self.log("✓ All tests passed!")
            return True
        else:
            self.log("✗ Some tests failed!")
            return False

def main():
    """Main test runner"""
    test_suite = MPITestSuite()
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if test_suite.rank == 0:
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()