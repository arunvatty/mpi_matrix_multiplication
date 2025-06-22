#!/bin/bash

# MPI Matrix Multiplication Benchmark Runner
# This script runs comprehensive benchmarks for both serial and MPI implementations

echo "MPI Matrix Multiplication Benchmark Suite"
echo "=========================================="

# Configuration
MATRIX_SIZES="100 200 400 800"
NUM_RUNS=3
PROCESS_COUNTS="2 4 8"

# Create directories for results
mkdir -p results
mkdir -p plots

echo "Starting benchmarks..."
echo

# Step 1: Run serial benchmark
echo "Step 1: Running serial matrix multiplication benchmark..."
python3 serial_matrix_multiplication.py \
    --sizes $MATRIX_SIZES \
    --runs $NUM_RUNS \
    --output results/serial_results.json

if [ $? -eq 0 ]; then
    echo "✓ Serial benchmark completed successfully"
else
    echo "✗ Serial benchmark failed"
    exit 1
fi

echo

# Step 2: Run MPI benchmarks for different process counts
echo "Step 2: Running MPI benchmarks..."

for procs in $PROCESS_COUNTS; do
    echo "Running MPI benchmark with $procs processes..."
    
    mpirun -np $procs python3 mpi_matrix_multiplication.py \
        --sizes $MATRIX_SIZES \
        --runs $NUM_RUNS \
        --verify \
        --output results/mpi_results_${procs}p.json
    
    if [ $? -eq 0 ]; then
        echo "✓ MPI benchmark with $procs processes completed successfully"
    else
        echo "✗ MPI benchmark with $procs processes failed"
        exit 1
    fi
    echo
done

# Step 3: Generate performance analysis
echo "Step 3: Generating performance analysis..."

MPI_FILES=""
for procs in $PROCESS_COUNTS; do
    MPI_FILES="$MPI_FILES results/mpi_results_${procs}p.json"
done

python3 performance_analyzer.py \
    --serial results/serial_results.json \
    --mpi $MPI_FILES \
    --output-dir plots \
    --report results/performance_report.txt

if [ $? -eq 0 ]; then
    echo "✓ Performance analysis completed successfully"
else
    echo "✗ Performance analysis failed"
    exit 1
fi

echo
echo "Benchmark suite completed successfully!"
echo "Results saved in: results/"
echo "Plots saved in: plots/"
echo
echo "Summary of files generated:"
echo "- results/serial_results.json       : Serial benchmark data"
for procs in $PROCESS_COUNTS; do
    echo "- results/mpi_results_${procs}p.json      : MPI benchmark data (${procs} processes)"
done
echo "- results/performance_report.txt    : Detailed performance analysis"
echo "- plots/execution_times.png         : Execution time comparison"
echo "- plots/speedup_analysis.png        : Speedup analysis"
echo "- plots/efficiency_analysis.png     : Efficiency analysis"
echo "- plots/scalability_analysis.png    : Scalability analysis"