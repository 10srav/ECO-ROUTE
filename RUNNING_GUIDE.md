# EcoRoute - Complete Running Guide

## System Requirements
- Ubuntu Linux
- Python 3.12
- Mininet (installed via: `sudo apt-get install mininet`)
- Root/sudo access (for Mininet)

## Current Running Status

### Active Components:
1. **Ryu SDN Controller** - Port 6653 (OpenFlow)
2. **Dashboard API** - Port 5000 (REST API)
3. **React Frontend** - NOT RUNNING (optional)

---

## Complete Setup & Run Commands

### 1. Initial Setup (One-time only)

```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
# Note: Ryu has been patched for Python 3.12 compatibility
pip install -r requirements.txt

# Install Mininet (requires sudo)
sudo apt-get update
sudo apt-get install mininet
```

### 2. Start the System

#### Option A: Using run.sh (Recommended)
```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE
chmod +x run.sh
./run.sh start
```

#### Option B: Manual Start (Step by Step)

**Terminal 1 - Start Controller:**
```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE
source venv/bin/activate
PYTHONPATH="$PWD" ryu-manager --observe-links controller/ecoroute_controller.py
```

**Terminal 2 - Start Dashboard API:**
```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE
source venv/bin/activate
PYTHONPATH="$PWD" python3 dashboard/flask_api.py --host 0.0.0.0 --port 5000
```

**Terminal 3 - Start Mininet Topology:**
```bash
sudo python3 /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE/topology/fat_tree_topo.py --k 4 --controller 127.0.0.1:6653
```

**Terminal 4 - Monitor Controller Logs:**
```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE
tail -f logs/controller.log
```

### 3. Test Network Connectivity

Once Mininet starts, in the Mininet CLI:

```bash
# Test all-to-all connectivity
pingall

# Test specific hosts
h1 ping -c 3 h2

# Check network topology
net

# List all nodes
nodes
```

### 4. Generate Traffic for Energy Optimization

In Mininet CLI:

```bash
# Simple bandwidth test
iperf h1 h2

# Longer duration test
h1 iperf -s &
h2 iperf -c h1 -t 60
```

### 5. Monitor System

#### Check API Health
```bash
curl http://localhost:5000/api/health
```

#### View Network Topology
```bash
curl -s http://localhost:5000/api/topology | python3 -m json.tool
```

#### Check Energy Statistics
```bash
curl -s http://localhost:5000/api/energy | python3 -m json.tool
```

#### View Traffic Predictions
```bash
curl -s http://localhost:5000/api/predictions | python3 -m json.tool
```

#### View Sleep/Wake Events
```bash
curl -s http://localhost:5000/api/events | python3 -m json.tool
```

### 6. Stop the System

**Stop Mininet (in Mininet terminal):**
```bash
exit
sudo mn -c
```

**Stop Controller and Dashboard:**
```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE

# Option A: Using run.sh
./run.sh stop

# Option B: Manual stop
kill $(cat logs/controller.pid)
kill $(cat logs/dashboard.pid)
```

**Or kill all processes:**
```bash
pkill -f "ryu-manager"
pkill -f "flask_api.py"
```

---

## React Frontend (Optional)

The React frontend is optional. To run it:

```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE/dashboard/frontend

# Install dependencies (first time only)
npm install

# Start development server
npm start
```

Access at: http://localhost:3000

---

## Troubleshooting

### Check if Services are Running
```bash
# Check controller
ps aux | grep ryu-manager

# Check dashboard API
ps aux | grep flask_api

# Check ports
ss -tlnp | grep -E "(5000|6653)"
```

### View Logs
```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE

# Controller logs
tail -f logs/controller.log

# Dashboard logs
tail -f logs/dashboard.log

# Filter for specific events
tail -f logs/controller.log | grep -E "(error|warning|sleep|wake)"
```

### Clean Up Mininet
If Mininet doesn't start properly:
```bash
sudo mn -c
sudo killall controller
```

### Restart Everything
```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE

# Stop all
pkill -f "ryu-manager"
pkill -f "flask_api"
sudo mn -c

# Wait a moment
sleep 3

# Start fresh
./run.sh start
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| GET /api/health | Health check |
| GET /api/topology | Network topology with all switches and links |
| GET /api/stats | Comprehensive network statistics |
| GET /api/energy | Power consumption and energy savings |
| GET /api/predictions | EWMA traffic predictions |
| GET /api/events | Sleep/wake event history |

---

## Expected Energy Savings

Under normal data center traffic patterns:
- **Energy Savings:** 25-35% vs ECMP baseline
- **Active Port Ratio:** <40% during low load
- **Packet Loss:** <0.1% during transitions
- **Latency Increase:** <5ms vs baseline

---

## Important Notes

1. **Python 3.12 Compatibility:** This installation includes patches to Ryu for Python 3.12 compatibility
2. **Mininet Requires Sudo:** Always run Mininet topology with sudo
3. **Controller Must Start First:** Start controller before Mininet topology
4. **Wait for Discovery:** Give the system 10-15 seconds after starting Mininet for topology discovery
5. **First Pingall May Fail:** The first pingall often has packet loss; run it 2-3 times

---

## File Locations

- **Controller Code:** `/home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE/controller/`
- **Dashboard API:** `/home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE/dashboard/`
- **Topology:** `/home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE/topology/`
- **Logs:** `/home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE/logs/`
- **Config:** `/home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE/config.yaml`
- **Virtual Env:** `/home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE/venv/`

---

## Quick Reference Commands

### Start Everything
```bash
cd /home/madavarapusaiharshavardhan/Eco-route/ECO-ROUTE
./run.sh start
# In new terminal:
sudo python3 topology/fat_tree_topo.py --k 4 --controller 127.0.0.1:6653
```

### Stop Everything
```bash
./run.sh stop
sudo mn -c
```

### Check Status
```bash
curl http://localhost:5000/api/health
ps aux | grep -E "(ryu|flask)"
```

### View Real-time Activity
```bash
tail -f logs/controller.log | grep "event"
```
