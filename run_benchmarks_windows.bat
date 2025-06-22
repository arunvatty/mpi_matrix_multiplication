@echo off
setlocal enabledelayedexpansion

echo MPI Matrix Multiplication Benchmark Suite - Windows
echo ====================================================

:: Configuration
set MATRIX_SIZES=100 200 400 800
set NUM_RUNS=3
set PROCESS_COUNTS=2 4 8

:: Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

:: Create directories for results
if not exist "results" mkdir results
if not exist "plots" mkdir plots

echo Starting benchmarks...
echo.

:: Step 1: Run serial benchmark
echo Step 1: Running serial matrix multiplication benchmark...
python serial_matrix_multiplication.py --sizes %MATRIX_SIZES% --runs %NUM_RUNS% --output results\serial_results.json

if errorlevel 1 (
    echo ✗ Serial benchmark failed
    pause
    exit /b 1
) else (
    echo ✓ Serial benchmark completed successfully
)

echo.

:: Step 2: Run MPI benchmarks for different process counts
echo Step 2: Running MPI benchmarks...

for %%p in (%PROCESS_COUNTS%) do (
    echo Running MPI benchmark with %%p processes...
    
    mpiexec -n %%p python mpi_matrix_multiplication.py --sizes %MATRIX_SIZES% --runs %NUM_RUNS% --verify --output results\mpi_results_%%pp.json
    
    if errorlevel 1 (
        echo ✗ MPI benchmark with %%p processes failed
        pause
        exit /b 1
    ) else (
        echo ✓ MPI benchmark with %%p processes completed successfully
    )
    echo.
)

:: Step 3: Generate performance analysis
echo Step 3: Generating performance analysis...

:: Build MPI files list
set MPI_FILES=
for %%p in (%PROCESS_COUNTS%) do (
    set MPI_FILES=!MPI_FILES! results\mpi_results_%%pp.json
)

python performance_analyzer.py --serial results\serial_results.json --mpi %MPI_FILES% --output-dir plots --report results\performance_report.txt

if errorlevel 1 (
    echo ✗ Performance analysis failed
    pause
    exit /b 1
) else (
    echo ✓ Performance analysis completed successfully
)

echo.
echo Benchmark suite completed successfully!
echo Results saved in: results\
echo Plots saved in: plots\
echo.
echo Summary of files generated:
echo - results\serial_results.json       : Serial benchmark data
for %%p in (%PROCESS_COUNTS%) do (
    echo - results\mpi_results_%%pp.json      : MPI benchmark data (%%p processes^)
)
echo - results\performance_report.txt    : Detailed performance analysis
echo - plots\execution_times.png         : Execution time comparison
echo - plots\speedup_analysis.png        : Speedup analysis
echo - plots\efficiency_analysis.png     : Efficiency analysis
echo - plots\scalability_analysis.png    : Scalability analysis
echo.
echo Press any key to open the results folder...
pause
explorer results