#!/bin/bash
# EcoRoute SDN Controller - Quick Start Script
# Usage: ./run.sh [command]
#   start      - Start the controller and dashboard
#   stop       - Stop all services
#   topology   - Start Mininet topology
#   test       - Run tests
#   benchmark  - Run benchmarks
#   clean      - Clean up

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Print banner
print_banner() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║   🌿 EcoRoute SDN Controller                                  ║"
    echo "║   Energy-Aware Dynamic Traffic Engineering                    ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check dependencies
check_deps() {
    echo -e "${BLUE}Checking dependencies...${NC}"

    # Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is required but not installed.${NC}"
        exit 1
    fi

    # pip
    if ! command -v pip3 &> /dev/null; then
        echo -e "${RED}pip3 is required but not installed.${NC}"
        exit 1
    fi

    echo -e "${GREEN}All dependencies found!${NC}"
}

# Install Python dependencies
install_deps() {
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    pip3 install -r requirements.txt
    echo -e "${GREEN}Dependencies installed!${NC}"
}

# Start the controller
start_controller() {
    echo -e "${BLUE}Starting EcoRoute Controller...${NC}"

    # Create logs directory
    mkdir -p logs

    # Check if Ryu is installed
    if ! python3 -c "import ryu" 2>/dev/null; then
        echo -e "${YELLOW}Ryu not found. Installing...${NC}"
        pip3 install ryu
    fi

    # Start controller in background
    echo -e "${GREEN}Starting Ryu controller...${NC}"
    PYTHONPATH="$PROJECT_DIR" ryu-manager \
        --observe-links \
        controller/ecoroute_controller.py \
        > logs/controller.log 2>&1 &

    CONTROLLER_PID=$!
    echo $CONTROLLER_PID > logs/controller.pid
    echo -e "${GREEN}Controller started with PID $CONTROLLER_PID${NC}"

    # Wait for controller to start
    sleep 3

    # Start dashboard
    echo -e "${GREEN}Starting Dashboard API...${NC}"
    PYTHONPATH="$PROJECT_DIR" python3 dashboard/flask_api.py \
        --host 0.0.0.0 \
        --port 5000 \
        > logs/dashboard.log 2>&1 &

    DASHBOARD_PID=$!
    echo $DASHBOARD_PID > logs/dashboard.pid
    echo -e "${GREEN}Dashboard started with PID $DASHBOARD_PID${NC}"

    echo ""
    echo -e "${GREEN}EcoRoute is running!${NC}"
    echo -e "  Controller:  ${BLUE}http://localhost:6653${NC} (OpenFlow)"
    echo -e "  Dashboard:   ${BLUE}http://localhost:5000${NC}"
    echo ""
    echo -e "Logs: ${YELLOW}logs/controller.log${NC}, ${YELLOW}logs/dashboard.log${NC}"
    echo -e "Stop with: ${YELLOW}./run.sh stop${NC}"
}

# Stop all services
stop_services() {
    echo -e "${BLUE}Stopping EcoRoute services...${NC}"

    # Kill controller
    if [ -f logs/controller.pid ]; then
        kill $(cat logs/controller.pid) 2>/dev/null || true
        rm logs/controller.pid
    fi

    # Kill dashboard
    if [ -f logs/dashboard.pid ]; then
        kill $(cat logs/dashboard.pid) 2>/dev/null || true
        rm logs/dashboard.pid
    fi

    # Kill any remaining ryu processes
    pkill -f "ryu-manager" 2>/dev/null || true
    pkill -f "flask_api.py" 2>/dev/null || true

    echo -e "${GREEN}All services stopped.${NC}"
}

# Start Mininet topology
start_topology() {
    echo -e "${BLUE}Starting Mininet Fat-Tree Topology...${NC}"

    if ! command -v mn &> /dev/null; then
        echo -e "${RED}Mininet is not installed.${NC}"
        echo "Install with: sudo apt-get install mininet"
        exit 1
    fi

    echo -e "${YELLOW}Note: This requires sudo privileges${NC}"
    sudo python3 topology/fat_tree_topo.py --k 4 --controller 127.0.0.1:6653
}

# Run tests
run_tests() {
    echo -e "${BLUE}Running tests...${NC}"

    PYTHONPATH="$PROJECT_DIR" python3 -m pytest tests/ -v --cov=controller --cov-report=html

    echo -e "${GREEN}Tests completed!${NC}"
    echo -e "Coverage report: ${YELLOW}htmlcov/index.html${NC}"
}

# Run benchmarks
run_benchmarks() {
    echo -e "${BLUE}Running benchmarks...${NC}"

    PYTHONPATH="$PROJECT_DIR" python3 benchmarks/traffic_test.py \
        --run-all \
        --duration 60 \
        --baseline \
        --export logs/benchmark_results.csv

    echo -e "${GREEN}Benchmarks completed!${NC}"
    echo -e "Results: ${YELLOW}logs/benchmark_results.csv${NC}"
}

# Docker commands
docker_build() {
    echo -e "${BLUE}Building Docker images...${NC}"
    docker-compose build
    echo -e "${GREEN}Docker images built!${NC}"
}

docker_start() {
    echo -e "${BLUE}Starting Docker containers...${NC}"
    docker-compose up -d
    echo -e "${GREEN}Docker containers started!${NC}"
    echo -e "  Controller:  ${BLUE}http://localhost:6653${NC}"
    echo -e "  Dashboard:   ${BLUE}http://localhost:5000${NC}"
}

docker_stop() {
    echo -e "${BLUE}Stopping Docker containers...${NC}"
    docker-compose down
    echo -e "${GREEN}Docker containers stopped!${NC}"
}

# Clean up
clean() {
    echo -e "${BLUE}Cleaning up...${NC}"

    stop_services

    rm -rf logs/*.log
    rm -rf logs/*.pid
    rm -rf logs/*.csv
    rm -rf __pycache__
    rm -rf controller/__pycache__
    rm -rf tests/__pycache__
    rm -rf .pytest_cache
    rm -rf htmlcov
    rm -rf .coverage

    echo -e "${GREEN}Cleaned up!${NC}"
}

# Show help
show_help() {
    print_banner
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start      Start the controller and dashboard (local)"
    echo "  stop       Stop all services"
    echo "  topology   Start Mininet fat-tree topology"
    echo "  test       Run unit tests"
    echo "  benchmark  Run traffic benchmarks"
    echo "  install    Install Python dependencies"
    echo "  docker-build   Build Docker images"
    echo "  docker-start   Start with Docker Compose"
    echo "  docker-stop    Stop Docker containers"
    echo "  clean      Clean up logs and temp files"
    echo "  help       Show this help message"
    echo ""
}

# Main
print_banner

case "${1:-help}" in
    start)
        check_deps
        start_controller
        ;;
    stop)
        stop_services
        ;;
    topology)
        start_topology
        ;;
    test)
        run_tests
        ;;
    benchmark)
        run_benchmarks
        ;;
    install)
        install_deps
        ;;
    docker-build)
        docker_build
        ;;
    docker-start)
        docker_start
        ;;
    docker-stop)
        docker_stop
        ;;
    clean)
        clean
        ;;
    help|*)
        show_help
        ;;
esac
