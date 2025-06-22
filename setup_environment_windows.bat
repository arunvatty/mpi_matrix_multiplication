@echo off
echo MPI Matrix Multiplication - Windows Environment Setup
echo =====================================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found: 
python --version

:: Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo Pip found:
pip --version

:: Install Microsoft MPI (if not already installed)
echo.
echo Checking for Microsoft MPI...
where mpiexec >nul 2>&1
if errorlevel 1 (
    echo Microsoft MPI not found. Please install it manually:
    echo 1. Download from: https://www.microsoft.com/en-us/download/details.aspx?id=57467
    echo 2. Install both msmpisetup.exe and msmpisdk.msi
    echo 3. Restart your command prompt
    echo.
    echo After installing MPI, run this script again.
    pause
    exit /b 1
) else (
    echo Microsoft MPI found:
    mpiexec -help | findstr "Microsoft MPI"
)

:: Create virtual environment (recommended)
echo.
echo Creating Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install Python dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Error installing dependencies. Trying alternative approach...
    echo Installing packages individually...
    
    pip install numpy
    pip install matplotlib
    pip install pandas
    pip install scipy
    
    :: Install mpi4py (may need special handling on Windows)
    echo Installing mpi4py...
    pip install mpi4py
    
    if errorlevel 1 (
        echo.
        echo mpi4py installation failed. Trying with conda...
        where conda >nul 2>&1
        if not errorlevel 1 (
            conda install -c conda-forge mpi4py
        ) else (
            echo Please install mpi4py manually:
            echo   1. Install Anaconda/Miniconda
            echo   2. Run: conda install -c conda-forge mpi4py
            echo   3. Or use pre-compiled wheels from: https://www.lfd.uci.edu/~gohlke/pythonlibs/
        )
    )
)

:: Test MPI4PY installation
echo.
echo Testing MPI4PY installation...
python -c "try: from mpi4py import MPI; print('✓ MPI4PY working correctly'); print('MPI Version:', MPI.Get_version()); except Exception as e: print('✗ MPI4PY test failed:', e)"

:: Create project directories
echo.
echo Creating project directories...
if not exist "results" mkdir results
if not exist "plots" mkdir plots
if not exist "docs" mkdir docs

echo.
echo Environment setup completed!
echo.
echo To activate the virtual environment in future sessions, run:
echo   venv\Scripts\activate.bat
echo.
echo Next steps:
echo 1. Run: python serial_matrix_multiplication.py
echo 2. Run: mpiexec -n 4 python mpi_matrix_multiplication.py
echo 3. Or use the automated batch file: run_benchmarks_windows.bat
echo.
pause