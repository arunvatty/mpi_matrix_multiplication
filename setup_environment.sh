#!/bin/bash

# MPI Environment Setup Script
# This script sets up the MPI development environment

echo "MPI Matrix Multiplication - Environment Setup"
echo "============================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages on Ubuntu/Debian
install_ubuntu() {
    echo "Installing MPI and dependencies on Ubuntu/Debian..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        openmpi-bin \
        openmpi-common \
        libopenmpi-dev \
        libhdf5-openmpi-dev
}

# Function to install packages on CentOS/RHEL/Fedora
install_redhat() {
    echo "Installing MPI and dependencies on CentOS/RHEL/Fedora..."
    if command_exists dnf; then
        sudo dnf install -y \
            gcc gcc-c++ \
            python3 python3-pip python3-devel \
            openmpi openmpi-devel \
            environment-modules
    else
        sudo yum install -y \
            gcc gcc-c++ \
            python3 python3-pip python3-devel \
            openmpi openmpi-devel \
            environment-modules
    fi
    
    # Load MPI module
    module load mpi/openmpi-x86_64 2>/dev/null || echo "Note: You may need to load MPI module manually"
}

# Function to install on macOS
install_macos() {
    echo "Installing MPI and dependencies on macOS..."
    if ! command_exists brew; then
        echo "Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    brew install open-mpi python3
}

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [ -f /etc/debian_version ]; then
        install_ubuntu
    elif [ -f /etc/redhat-release ]; then
        install_redhat
    else
        echo "Unsupported Linux distribution. Please install MPI manually."
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    install_macos
else
    echo "Unsupported operating system: $OSTYPE"
    echo "Please install MPI manually."
    exit 1
fi

# Check if MPI is properly installed
echo
echo "Checking MPI installation..."
if command_exists mpirun; then
    echo "✓ mpirun found: $(which mpirun)"
    mpirun --version | head -1
else
    echo "✗ mpirun not found in PATH"
    echo "You may need to add MPI to your PATH or load the MPI module"
    exit 1
fi

if command_exists mpicc; then
    echo "✓ mpicc found: $(which mpicc)"
else
    echo "✗ mpicc not found in PATH"
fi

# Install Python dependencies
echo
echo "Installing Python dependencies..."
if command_exists pip3; then
    pip3 install --user -r requirements.txt
elif command_exists pip; then
    pip install --user -r requirements.txt
else
    echo "✗ pip not found. Please install pip first."
    exit 1
fi

# Test MPI4PY installation
echo
echo "Testing MPI4PY installation..."
python3 -c "
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print(f'✓ MPI4PY working correctly')
    print(f'  MPI Version: {MPI.Get_version()}')
    print(f'  Available processes: {comm.Get_size()}')
except ImportError as e:
    print(f'✗ MPI4PY import failed: {e}')
    exit(1)
except Exception as e:
    print(f'✗ MPI4PY test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "✓ MPI4PY installation successful"
else
    echo "✗ MPI4PY installation failed"
    exit 1
fi

# Create project structure
echo
echo "Creating project structure..."
mkdir -p results
mkdir -p plots
mkdir -p docs

# Make scripts executable
chmod +x run_benchmarks.sh
chmod +x setup_environment.sh

echo
echo "Environment setup completed successfully!"
echo
echo "Next steps:"
echo "1. Run the benchmark suite: ./run_benchmarks.sh"
echo "2. Or run individual components:"
echo "   - Serial benchmark: python3 serial_matrix_multiplication.py"
echo "   - MPI benchmark: mpirun -np 4 python3 mpi_matrix_multiplication.py"
echo "   - Performance analysis: python3 performance_analyzer.py"
echo
echo "For cloud deployment, see the deployment guide in the documentation."