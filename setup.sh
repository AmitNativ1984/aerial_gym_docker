#!/bin/bash
# =============================================================================
# Aerial Gym Docker Setup Script
# =============================================================================
# This script sets up the directory structure and checks prerequisites
# Run this AFTER downloading Isaac Gym Preview 4
# =============================================================================

set -e

echo "=========================================="
echo "Aerial Gym Docker Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA driver not found. Please install NVIDIA drivers first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ NVIDIA driver found${NC}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found. Please install Docker first.${NC}"
    echo "Run: curl -fsSL https://get.docker.com | sh"
    exit 1
fi
echo -e "${GREEN}✓ Docker found${NC}"

# Check NVIDIA Container Toolkit
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo -e "${YELLOW}WARNING: NVIDIA Container Toolkit may not be configured.${NC}"
    echo "Testing GPU access in Docker..."
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${RED}ERROR: Cannot access GPU in Docker. Please install nvidia-container-toolkit.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ NVIDIA Container Toolkit working${NC}"

# -----------------------------------------------------------------------------
# Check for Isaac Gym package
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/6] Checking for Isaac Gym package...${NC}"

ISAAC_GYM_TAR=""
if [ -f "IsaacGym_Preview_4_Package.tar.gz" ]; then
    ISAAC_GYM_TAR="IsaacGym_Preview_4_Package.tar.gz"
elif [ -f "$HOME/Downloads/IsaacGym_Preview_4_Package.tar.gz" ]; then
    ISAAC_GYM_TAR="$HOME/Downloads/IsaacGym_Preview_4_Package.tar.gz"
    echo "Found Isaac Gym in ~/Downloads/, copying..."
    cp "$ISAAC_GYM_TAR" .
    ISAAC_GYM_TAR="IsaacGym_Preview_4_Package.tar.gz"
fi

if [ -z "$ISAAC_GYM_TAR" ]; then
    echo -e "${RED}ERROR: Isaac Gym package not found!${NC}"
    echo ""
    echo "Please download Isaac Gym Preview 4 from:"
    echo "  https://developer.nvidia.com/isaac-gym"
    echo ""
    echo "Then either:"
    echo "  1. Place IsaacGym_Preview_4_Package.tar.gz in this directory"
    echo "  2. Or leave it in ~/Downloads/"
    echo ""
    echo "Then run this script again."
    exit 1
fi
echo -e "${GREEN}✓ Isaac Gym package found: $ISAAC_GYM_TAR${NC}"

# -----------------------------------------------------------------------------
# Extract Isaac Gym
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/6] Extracting Isaac Gym...${NC}"

if [ -d "isaacgym" ]; then
    echo "isaacgym directory already exists, skipping extraction"
else
    tar -xzf "$ISAAC_GYM_TAR"
    echo -e "${GREEN}✓ Isaac Gym extracted${NC}"
fi

# -----------------------------------------------------------------------------
# Create directory structure
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/6] Creating directory structure...${NC}"

mkdir -p outputs logs checkpoints custom_code

echo -e "${GREEN}✓ Directories created${NC}"

# -----------------------------------------------------------------------------
# Create/verify Dockerfile and docker-compose.yml
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/6] Checking Docker files...${NC}"

if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}ERROR: Dockerfile not found in current directory${NC}"
    echo "Please ensure Dockerfile is in this directory"
    exit 1
fi
echo -e "${GREEN}✓ Dockerfile found${NC}"

if [ ! -f "docker-compose.yml" ]; then
    echo -e "${YELLOW}WARNING: docker-compose.yml not found (optional)${NC}"
else
    echo -e "${GREEN}✓ docker-compose.yml found${NC}"
fi

# -----------------------------------------------------------------------------
# Build Docker image
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/6] Building Docker image...${NC}"
echo "This may take 15-30 minutes on first build..."
echo ""

read -p "Do you want to build the Docker image now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "docker-compose.yml" ]; then
        docker compose build
    else
        docker build -t aerial-gym:latest .
    fi
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo "Skipping build. Run 'docker compose build' or 'docker build -t aerial-gym:latest .' later."
fi

# -----------------------------------------------------------------------------
# Print summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "  $(pwd)/"
echo "  ├── Dockerfile"
echo "  ├── docker-compose.yml"
echo "  ├── isaacgym/"
echo "  ├── outputs/          <- Training outputs will be saved here"
echo "  ├── logs/             <- Logs will be saved here"
echo "  ├── checkpoints/      <- Model checkpoints"
echo "  └── custom_code/      <- Put your SNN code here"
echo ""
echo "To run the container:"
echo "  Headless:  docker compose run aerial-gym"
echo "  With GUI:  xhost +local:docker && docker compose run aerial-gym-gui"
echo ""
echo "Inside the container, test with:"
echo "  cd /opt/isaacgym/python/examples && python joint_monkey.py --headless"
echo "  cd /workspace/aerial_gym_simulator && python examples/position_control_example.py --headless"
echo ""
