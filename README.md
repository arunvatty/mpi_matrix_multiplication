# MPI-Based Distributed Matrix Multiplication

A comprehensive implementation and benchmarking suite for distributed matrix multiplication using MPI (Message Passing Interface) in Python.

## 🎯 Project Overview

This project implements and evaluates the performance of matrix multiplication across multiple nodes using MPI, focusing on:
- **Distributed computation** with data partitioning strategies
- **Inter-process communication** optimization
- **Performance metrics** and scalability analysis
- **Comprehensive benchmarking** against serial implementations

## 📁 Project Structure

```
mpi-matrix-multiplication/
├── serial_matrix_multiplication.py    # Serial benchmark implementation
├── mpi_matrix_multiplication.py       # MPI distributed implementation
├── performance_analyzer.py            # Performance analysis and visualization
├── test_mpi_implementation.py         # Comprehensive test suite
├── requirements.txt                   # Python dependencies
├── setup_environment.sh               # Environment setup script
├── run_benchmarks.sh                  # Automated benchmark runner
├── results/                           # Benchmark results directory
├── plots/                             # Generated performance plots
└── docs/                              # Documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repository-url>
cd mpi-matrix-multiplication

# Make scripts executable
chmod +x setup_environment.sh run_benchmarks.sh

# Setup MPI environment (installs MPI and Python dependencies)
./setup_environment.sh
```

### 2. Run Benchmarks

```bash
# Run complete benchmark suite
./run_benchmarks.sh

# Or run individual components:

# Serial benchmark
python3 serial_matrix_multiplication.py --sizes 100 200 400 --runs 3

# MPI benchmark (example with 4 processes)
mpirun -np 4 python3 mpi_matrix_multiplication.py --sizes 100 200 400 --runs 3 --verify

# Performance analysis
python3 performance_analyzer.py --serial results/serial_results.json --mpi results/mpi_results_4p.json
```

### 3. Run Tests

```bash
# Test with different numbers of processes
mpirun -np 1 python3 test_mpi_implementation.py
mpirun -np 4 python3 test_mpi_implementation.py
mpirun -np 8 python3 test_mpi_implementation.py
```

## 🔧 Implementation Details

### Matrix Multiplication Algorithm

The project implements **row-wise partitioning** strategy:

1. **Data Distribution**: Matrix A is partitioned row-wise among processes
2. **Broadcasting**: Matrix B is broadcast to all processes
3. **Local Computation**: Each process computes its portion of the result
4. **Result Gathering**: Results are collected at the root process

### Key Features

- **Scalable Design**: Handles matrices larger than the number of processes
- **Load Balancing**: Distributes rows evenly with remainder handling
- **Error Handling**: Comprehensive error checking and validation
- **Performance Monitoring**: Detailed timing and efficiency metrics

### Communication Pattern

```
Process 0: Rows 0 to (n/p)-1 of Matrix A
Process 1: Rows n/p to (2n/p)-1 of Matrix A
...
Process p-1: Remaining rows of Matrix A

All processes receive complete Matrix B via broadcast
```

## 📊 Performance Analysis

The performance analyzer generates:

1. **Execution Time Comparison**: Serial vs MPI performance
2. **Speedup Analysis**: Actual vs theoretical speedup
3. **Efficiency Analysis**: Parallel efficiency metrics
4. **Scalability Analysis**: Performance across different process counts

### Sample Results

```
Matrix Size: 400x400
Serial Time: 2.1500 seconds
MPI (4 processes):
  Time: 0.6250 seconds
  Speedup: 3.44x
  Efficiency: 0.86 (86%)
```

## 🧪 Testing

The test suite includes:

- **Correctness Testing**: Verification against serial implementation
- **Edge Case Testing**: Small matrices, single process scenarios
- **Data Type Testing**: Different numerical precisions
- **Performance Consistency**: Multiple run validation
- **Communication Testing**: MPI operation verification

## 📈 Benchmarking

### Default Benchmark Configuration

- **Matrix Sizes**: 100×100, 200×200, 400×400, 800×800
- **Process Counts**: 2, 4, 8 processes
- **Runs per Test**: 3 iterations for statistical significance
- **Metrics**: Execution time, speedup, efficiency

### Performance Expectations

Typical speedup results:
- **2 processes**: 1.6-1.8x speedup
- **4 processes**: 2.8-3.5x speedup
- **8 processes**: 4.0-6.0x speedup (depends on matrix size)

## 🖥️ System Requirements

### Software Requirements
- **Python**: 3.7 or higher
- **MPI Implementation**: OpenMPI or MPICH
- **Libraries**: numpy, mpi4py, matplotlib, pandas

### Hardware Recommendations
- **Minimum**: 2 CPU cores, 4GB RAM
- **Optimal**: 4+ CPU cores, 8GB+ RAM
- **For large matrices**: 16GB+ RAM recommended

## 🌐 Cloud Deployment

### AWS EC2 Example

```bash
# Launch multiple EC2 instances
# Install MPI cluster setup
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# Configure SSH keys for passwordless access
# Create hostfile with instance IPs
# Run: mpirun -np 8 --hostfile hostfile python3 mpi_matrix_multiplication.py
```

### Google Cloud Platform

```bash
# Use Compute Engine with MPI
# Or Google Kubernetes Engine with MPI Operator
# Configure persistent disks for large datasets
```

## 📝 Documentation

### Code Documentation

Each module includes comprehensive docstrings:

```python
def mpi_matrix_multiply(size, comm, rank, num_procs):
    """
    Distributed matrix multiplication using MPI
    
    Args:
        size (int): Matrix dimension (square matrices)
        comm: MPI communicator
        rank (int): Process rank
        num_procs (int): Total number of processes
    
    Returns:
        numpy.ndarray: Result matrix (only on root process)
    """
```

### Algorithm Complexity

- **Serial**: O(n³) time, O(n²) space
- **MPI**: O(n³/p) time per process, O(n²) space per process
- **Communication**: O(n²) for broadcast, O(n²/p) for gather

## 🐛 Troubleshooting

### Common Issues

1. **MPI Import Error**
   ```bash
   # Reinstall mpi4py
   pip3 uninstall mpi4py
   pip3 install mpi4py
   ```

2. **Process Communication Timeout**
   ```bash
   # Increase timeout or reduce matrix size
   export OMPI_MCA_btl_tcp_if_include=lo,eth0
   ```

3. **Memory Issues**
   ```bash
   # Reduce matrix size or increase swap space
   # Monitor with: free -h
   ```

### Performance Optimization Tips

1. **Matrix Size**: Use sizes divisible by process count
2. **Process Count**: Match available CPU cores
3. **Memory**: Ensure sufficient RAM for largest matrices
4. **Network**: Use high-speed interconnects for clusters

## 📋 Deliverables Checklist

- ✅ **MPI-based distributed matrix multiplication code**
- ✅ **Performance metrics and timing system**
- ✅ **Benchmarking against serial implementation**
- ✅ **Scalability testing across multiple process counts**
- ✅ **Comprehensive documentation and setup guides**
- ✅ **Test suite for correctness verification**
- ✅ **Automated benchmark runner**
- ✅ **Performance visualization and analysis tools**

## 🔍 Results Analysis

The benchmark generates detailed analysis including:

- **Performance Report**: `results/performance_report.txt`
- **Execution Time Plot**: `plots/execution_times.png`
- **Speedup Analysis**: `plots/speedup_analysis.png`
- **Efficiency Analysis**: `plots/efficiency_analysis.png`
- **Scalability Analysis**: `plots/scalability_analysis.png`

## 📚 References

1. MPI Forum. "MPI: A Message-Passing Interface Standard"
2. Pacheco, P. "Parallel Programming with MPI"
3. Foster, I. "Designing and Building Parallel Programs"

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Run the test suite
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This implementation is designed for educational and research purposes. For production use, consider additional optimizations such as block-wise partitioning, advanced communication patterns, and specialized linear algebra libraries.