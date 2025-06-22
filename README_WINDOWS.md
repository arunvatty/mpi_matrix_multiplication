# MPI Matrix Multiplication - Windows Setup Guide

## 🖥️ Windows-Specific Installation Instructions

### Prerequisites

1. **Python 3.7+** installed with pip
   - Download from: https://www.python.org/downloads/
   - ⚠️ **Important**: Check "Add Python to PATH" during installation

2. **Microsoft MPI** (required for mpi4py)
   - Download from: https://www.microsoft.com/en-us/download/details.aspx?id=57467
   - Install **both** files:
     - `msmpisetup.exe` (Microsoft MPI redistributable)
     - `msmpisdk.msi` (Microsoft MPI SDK)

### Quick Start (Windows)

1. **Download/Clone the project**
   ```cmd
   git clone <your-repository-url>
   cd mpi-matrix-multiplication
   ```

2. **Run the Windows setup script**
   ```cmd
   setup_environment_windows.bat
   ```

3. **Run the benchmark suite**
   ```cmd
   run_benchmarks_windows.bat
   ```

### Manual Setup (Alternative)

If the automated scripts don't work, follow these manual steps:

#### Step 1: Install Microsoft MPI
```cmd
# Download and install from Microsoft:
# https://www.microsoft.com/en-us/download/details.aspx?id=57467

# Verify installation
mpiexec -help
```

#### Step 2: Install Python Dependencies
```cmd
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install numpy matplotlib pandas scipy

# Install mpi4py (may require special handling)
pip install mpi4py
```

#### Step 3: Test Installation
```cmd
# Test MPI
mpiexec -n 2 python -c "from mpi4py import MPI; print(f'Process {MPI.COMM_WORLD.Get_rank()} of {MPI.COMM_WORLD.Get_size()}')"

# Expected output:
# Process 0 of 2
# Process 1 of 2
```

### Running Individual Components

#### Serial Benchmark
```cmd
python serial_matrix_multiplication.py --sizes 100 200 400 --runs 3
```

#### MPI Benchmark
```cmd
# Single process (serial-like)
mpiexec -n 1 python mpi_matrix_multiplication.py --verify

# Multiple processes
mpiexec -n 4 python mpi_matrix_multiplication.py --sizes 100 200 400 --runs 3 --verify
```

#### Performance Analysis
```cmd
python performance_analyzer.py --serial results\serial_results.json --mpi results\mpi_results_4p.json
```

#### Test Suite
```cmd
mpiexec -n 4 python test_mpi_implementation.py
```

## 🔧 Troubleshooting Windows Issues

### Common Problems and Solutions

#### 1. "Python is not recognized as an internal or external command"
**Solution**: Add Python to PATH
```cmd
# Find Python installation
where python

# If not found, reinstall Python with "Add to PATH" checked
# Or manually add Python directory to PATH environment variable
```

#### 2. "mpiexec is not recognized"
**Solution**: Install Microsoft MPI
- Download both msmpisetup.exe and msmpisdk.msi
- Restart command prompt after installation
- Verify with: `mpiexec -help`

#### 3. "ImportError: No module named 'mpi4py'"
**Solutions**:
```cmd
# Option 1: Use conda (if you have Anaconda/Miniconda)
conda install -c conda-forge mpi4py

# Option 2: Use pre-compiled wheels
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#mpi4py
# Install with: pip install downloaded_wheel_file.whl

# Option 3: Reinstall with proper compiler
pip uninstall mpi4py
pip install mpi4py
```

#### 4. "Access denied" or Permission Errors
**Solution**: Run Command Prompt as Administrator
```cmd
# Right-click Command Prompt -> "Run as administrator"
# Then run the setup scripts
```

#### 5. Virtual Environment Issues
```cmd
# If venv doesn't work, try:
python -m pip install virtualenv
virtualenv venv
venv\Scripts\activate.bat
```

### Alternative: Using Anaconda (Recommended for Windows)

If you encounter persistent issues, Anaconda provides better Windows compatibility:

```cmd
# Install Anaconda from: https://www.anaconda.com/products/distribution

# Create conda environment
conda create -n mpi-project python=3.9
conda activate mpi-project

# Install packages
conda install numpy matplotlib pandas scipy
conda install -c conda-forge mpi4py

# Install Microsoft MPI separately (still required)
```

## 📁 Windows File Structure

```
mpi-matrix-multiplication\
├── README_WINDOWS.md                   # This file
├── setup_environment_windows.bat       # Windows setup script
├── run_benchmarks_windows.bat          # Windows benchmark runner
├── serial_matrix_multiplication.py     # Serial implementation
├── mpi_matrix_multiplication.py        # MPI implementation
├── performance_analyzer.py             # Analysis tool
├── test_mpi_implementation.py          # Test suite
├── requirements.txt                    # Python dependencies
├── venv\                               # Virtual environment (created by setup)
├── results\                            # Benchmark results
│   ├── serial_results.json
│   ├── mpi_results_2p.json
│   └── performance_report.txt
└── plots\                              # Generated plots
    ├── execution_times.png
    ├── speedup_analysis.png
    └── efficiency_analysis.png
```

## 🎯 Expected Windows Performance

Windows performance may be slightly different from Linux due to:
- Different MPI implementation (Microsoft MPI vs OpenMPI)
- Windows process management overhead
- Antivirus software interference

**Typical results on Windows:**
- **2 processes**: 1.4-1.7x speedup
- **4 processes**: 2.5-3.2x speedup
- **8 processes**: 3.5-5.5x speedup

## 📸 Screenshots to Capture for Documentation

1. **Setup verification**: Run `setup_environment_windows.bat` and capture success message
2. **MPI test**: Run `mpiexec -n 2 python -c "from mpi4py import MPI; print('Working')"` 
3. **Benchmark execution**: Run `run_benchmarks_windows.bat` and capture output
4. **Performance plots**: Open generated PNG files in `plots\` folder
5. **Test results**: Run `mpiexec -n 4 python test_mpi_implementation.py`

## 💡 Tips for Windows Users

1. **Use Windows Terminal** (available from Microsoft Store) for better command-line experience
2. **Disable antivirus temporarily** during benchmark runs for better performance
3. **Close other applications** to ensure accurate performance measurements
4. **Use PowerShell** as an alternative to Command Prompt if needed
5. **Consider WSL2** (Windows Subsystem for Linux) for Linux-like environment

## 🔄 Converting to Linux Commands (if needed)

| Windows | Linux |
|---------|-------|
| `dir` | `ls` |
| `type file.txt` | `cat file.txt` |
| `copy` | `cp` |
| `del` | `rm` |
| `mkdir` | `mkdir` |
| `cd` | `cd` |
| `mpiexec` | `mpirun` |
| `venv\Scripts\activate.bat` | `source venv/bin/activate` |
| `results\file.json` | `results/file.json` |

This Windows-specific guide ensures you can successfully run the MPI matrix multiplication project on your Windows machine!