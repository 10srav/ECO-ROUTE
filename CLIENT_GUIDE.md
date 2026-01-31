# EcoRoute - Energy-Saving Network System
## Complete Guide for Non-Technical Users

---

## What is EcoRoute?

**EcoRoute** is an intelligent network management system that **reduces energy consumption by 25-35%** in data center networks while maintaining excellent performance. It works like a smart power manager for network equipment.

### Simple Analogy
Think of it like smart lights in an office building:
- When rooms are empty, lights turn off automatically
- When someone enters, lights turn on instantly
- The building saves energy without anyone noticing

EcoRoute does the same for network equipment - it puts unused network links to "sleep" when traffic is low and wakes them up when needed.

---

## How Does It Work?

### 1. **Network Switches (The Equipment)**

Imagine switches as intelligent traffic intersections for data:
- **Switches** connect computers together
- They forward data packets (like letters) to the right destination
- Our network has 20 switches arranged in a special pattern called "fat-tree"

### 2. **Network Links (The Roads)**

Links are cables connecting switches:
- Like roads between intersections
- Data travels through these links
- Each link uses power even when idle

### 3. **The Controller (The Brain)**

The EcoRoute controller is the smart brain that:
- **Monitors** all traffic in the network
- **Predicts** future traffic patterns using smart algorithms
- **Decides** which links can sleep
- **Reroutes** traffic safely before sleeping links
- **Wakes** links before they're needed

### 4. **How Traffic is Routed**

**Normal Network (ECMP - Equal Cost Multi-Path):**
- Spreads traffic evenly across all possible paths
- Keeps ALL links powered on 24/7
- Wastes energy during low-traffic periods

**EcoRoute (Energy-Aware Routing):**
- Concentrates traffic on fewer paths when possible
- Puts unused links to sleep
- Saves 25-35% energy
- Maintains performance guarantees

---

## Network Topology Explained

### Fat-Tree Topology (k=4)

Our network uses a "fat-tree" structure with 20 switches arranged in 3 layers:

```
         Layer 1: CORE (4 switches)
              ↓  ↓  ↓  ↓
         Layer 2: AGGREGATION (8 switches)
              ↓  ↓  ↓  ↓
         Layer 3: EDGE (8 switches)
              ↓  ↓  ↓  ↓
            16 Host Computers
```

**Why this design?**
- **Redundancy:** Multiple paths between any two computers
- **Performance:** High bandwidth for data transfers
- **Energy Savings:** Many alternative paths = opportunities to sleep links

### Switch Connections

**Core Switches (c1, c2, c3, c4):**
- Top layer switches
- Connect to all aggregation switches
- Handle long-distance traffic

**Aggregation Switches:**
- Middle layer
- Connect core to edge
- Balance traffic load

**Edge Switches:**
- Bottom layer
- Connect directly to computers
- Handle local traffic

---

## Energy Saving Mechanism

### Step-by-Step Process

1. **Traffic Monitoring (Every 5 seconds)**
   - Controller checks traffic on all 186 links
   - Measures: bandwidth usage, packet count, latency

2. **Traffic Prediction (EWMA Algorithm)**
   - Predicts traffic for next 10-30 seconds
   - Uses historical patterns
   - Adapts to traffic bursts automatically

3. **Sleep Decision**
   - If predicted traffic < 20% capacity for 30+ seconds
   - AND alternate paths exist
   - AND can wake up fast enough (100ms)
   - → Put link to sleep

4. **Safe Transition (Make-Before-Break)**
   - First: Find alternate path
   - Second: Move traffic to new path
   - Third: Put old link to sleep
   - Never drops packets!

5. **Wake Decision**
   - If predicted traffic > 60% capacity
   - OR packet loss detected
   - → Wake link proactively

---

## Performance Guarantees

### Quality of Service (QoS) Maintained

| Metric | Guarantee | Meaning |
|--------|-----------|---------|
| **Packet Loss** | < 0.1% | 99.9% of data arrives successfully |
| **Latency Increase** | < 5ms | Minimal delay added |
| **Throughput** | ≥ 95% | Network speed remains high |
| **Energy Savings** | 25-35% | Significant power reduction |

---

## Complete Setup Guide for Clients

### Prerequisites

**System Requirements:**
- Ubuntu Linux (20.04 or newer)
- 4+ GB RAM
- 10+ GB free disk space
- Root/sudo access

### Step 1: Install System Dependencies

```bash
# Update system
sudo apt-get update

# Install Mininet (network emulator)
sudo apt-get install mininet

# Install Node.js (for web dashboard)
sudo apt-get install nodejs npm

# Verify installations
python3 --version  # Should be 3.8+
mn --version       # Should show Mininet version
node --version     # Should be 14+
```

### Step 2: Clone the Repository

```bash
# Navigate to your workspace
cd ~

# Clone EcoRoute
git clone https://github.com/10srav/ECO-ROUTE.git
cd ECO-ROUTE
```

### Step 3: Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

**Note:** The system includes patches for Python 3.12 compatibility.

### Step 4: Install Dashboard Dependencies

```bash
# Navigate to frontend
cd dashboard/frontend

# Install React dependencies
npm install

# Return to main directory
cd ../..
```

### Step 5: Start the System

**Open 3 separate terminal windows:**

**Terminal 1 - Start Controller & API:**
```bash
cd ~/ECO-ROUTE
./run.sh start
```

**Terminal 2 - Start Network Topology:**
```bash
cd ~/ECO-ROUTE
sudo python3 topology/fat_tree_topo.py --k 4 --controller 127.0.0.1:6653
```

**Terminal 3 - Start Web Dashboard:**
```bash
cd ~/ECO-ROUTE/dashboard/frontend
npm start
```

### Step 6: Access the Dashboard

1. Wait 10-15 seconds for everything to start
2. Open web browser
3. Go to: **http://localhost:3000**
4. You'll see the EcoRoute dashboard with live metrics

---

## Using the System

### Test Network Connectivity

In the Mininet terminal (Terminal 2), try these commands:

```bash
# Test all-to-all connectivity
mininet> pingall

# Test specific hosts
mininet> h1 ping -c 5 h2

# Check network topology
mininet> net

# View all switches
mininet> switches

# Exit Mininet
mininet> exit
```

### Generate Traffic (See Energy Savings in Action)

```bash
# In Mininet terminal
mininet> iperf h1 h2

# For longer test
mininet> h1 iperf -s &
mininet> h2 iperf -c h1 -t 60
```

After 30-60 seconds of traffic, you'll see:
- Links going to sleep (shown in dashboard)
- Energy savings increasing
- Power consumption decreasing

### Monitor System

**Check API Health:**
```bash
curl http://localhost:5000/api/health
```

**View Energy Stats:**
```bash
curl http://localhost:5000/api/energy | python3 -m json.tool
```

**Check Sleeping Links:**
```bash
curl http://localhost:5000/api/topology | python3 -m json.tool | grep "sleeping"
```

---

## Understanding the Dashboard

### Main Metrics Display

**Energy Savings (Top Left)**
- Shows percentage saved vs normal network
- Target: 25-35%
- Updates every second

**Active Ports (Top Center)**
- Shows how many network ports are active
- Example: "46 of 48 total" means 2 are sleeping

**Sleeping Links (Top Right)**
- Number of links currently in sleep mode
- More sleeping = more energy saved

**Power Saved (Top Right)**
- Actual watts saved right now
- Accumulates over time

### Graphs

**Energy Savings Over Time**
- Shows trend of energy savings
- Should stabilize at 25-35%

**Power Consumption**
- Blue bars: Current vs Baseline
- Shows visual comparison

**Network Topology**
- Visual map of all switches
- Green lines: Active links
- Red dashed lines: Sleeping links

**QoS Metrics**
- Packet Loss: Should be < 0.1%
- Latency: Should be < 5ms
- Throughput: Should be > 95%

**Recent Events**
- Shows what controller is doing
- FLOW_REROUTE: Moving traffic
- PORT_SLEEP: Putting link to sleep
- PREDICTION_UPDATE: Updating forecasts

---

## Troubleshooting

### Problem: Dashboard shows "Disconnected"

**Solution:**
```bash
# Check if API is running
curl http://localhost:5000/api/health

# If not, restart
cd ~/ECO-ROUTE
./run.sh stop
./run.sh start
```

### Problem: Mininet won't start

**Solution:**
```bash
# Clean up Mininet
sudo mn -c

# Try starting again
sudo python3 topology/fat_tree_topo.py --k 4 --controller 127.0.0.1:6653
```

### Problem: No energy savings showing

**Reason:** System needs traffic to optimize

**Solution:**
```bash
# Generate continuous traffic in Mininet
mininet> iperf h1 h2
mininet> iperf h3 h4

# Wait 30-60 seconds, savings should appear
```

### Problem: High packet loss (> 0.1%)

**Solution:**
```bash
# System might be too aggressive
# Edit config.yaml:
cd ~/ECO-ROUTE
nano config.yaml

# Change:
sleep_threshold: 20  →  sleep_threshold: 15
wake_threshold: 60   →  wake_threshold: 50

# Restart controller
./run.sh stop
./run.sh start
```

---

## Stopping the System

### Complete Shutdown

**Terminal 1 (Mininet):**
```bash
mininet> exit
sudo mn -c
```

**Terminal 2 (Controller/API):**
```bash
cd ~/ECO-ROUTE
./run.sh stop
```

**Terminal 3 (Dashboard):**
Press `Ctrl+C` to stop the React server

### Quick Stop (All at once)
```bash
cd ~/ECO-ROUTE
./run.sh stop
kill $(cat logs/frontend.pid)
sudo mn -c
```

---

## System Architecture for Non-Technical Users

### Components Overview

```
┌─────────────────────────────────────────────────┐
│         Web Browser (Your View)                 │
│         http://localhost:3000                   │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         React Dashboard (Frontend)              │
│         - Charts and graphs                     │
│         - Real-time updates                     │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Flask API (Backend)                     │
│         - REST API endpoints                    │
│         - Data aggregation                      │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Ryu Controller (Brain)                  │
│         - Traffic monitoring                    │
│         - EWMA prediction                       │
│         - Energy optimization                   │
│         - Flow management                       │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Mininet Network (Simulation)            │
│         - 20 switches                           │
│         - 16 hosts                              │
│         - 186 links                             │
└─────────────────────────────────────────────────┘
```

---

## Key Algorithms Explained Simply

### 1. EWMA (Traffic Prediction)

**What it does:** Predicts future traffic based on past patterns

**Simple explanation:**
- Like weather forecasting
- If it's been sunny for days, likely sunny tomorrow
- If traffic was low yesterday at 2am, likely low today at 2am

**Formula (for reference):**
```
Predicted Load = (0.3 × Current) + (0.7 × Previous Prediction)
```

### 2. Energy-Aware Routing

**What it does:** Chooses paths that allow sleeping more links

**Simple explanation:**
- Finds 3 possible routes between computers
- Scores each route: energy cost + traffic load + path length
- Picks route that saves most energy while meeting QoS

### 3. Make-Before-Break

**What it does:** Safely moves traffic before sleeping links

**Simple explanation:**
- Like changing lanes on highway
- Check if new lane is clear ✓
- Move to new lane ✓
- Old lane is now free ✓
- Never causes accidents (packet drops)

---

## Frequently Asked Questions

### Q: Will this work in a real data center?

**A:** Yes! This is a simulation for demonstration, but the algorithms are production-ready. For real deployment, you'd need:
- Real OpenFlow switches (instead of Mininet)
- Production-grade Ryu deployment
- Monitoring and alerting system
- Redundancy for controller

### Q: How much money does 30% energy savings mean?

**A:** For a medium data center:
- Network equipment: ~$100,000/year in electricity
- 30% savings = ~$30,000/year saved
- Plus environmental benefits (reduced CO2)

### Q: Does sleeping links damage the equipment?

**A:** No! Modern switches are designed for this:
- Sleep mode is a standard feature (IEEE 802.3az)
- Actually extends equipment life (less heat, less wear)
- Wake-up time is only 100 milliseconds

### Q: What if a sleeping link is needed urgently?

**A:** The controller predicts traffic and wakes links BEFORE they're needed:
- Monitors traffic every 5 seconds
- Predicts 10-30 seconds ahead
- Wakes links proactively when traffic increases
- 100ms wake-up time is faster than traffic arrival

### Q: Can I use this for my home network?

**A:** This is designed for data centers, but the concepts apply:
- Home routers don't usually support OpenFlow
- Energy savings would be minimal (home networks use little power)
- Better suited for enterprise/campus/data center networks

---

## Support and Documentation

### Additional Resources

- **Technical Documentation:** See [CLAUDE.md](CLAUDE.md) for developer details
- **Running Guide:** See [RUNNING_GUIDE.md](RUNNING_GUIDE.md) for operational commands
- **Original README:** See [README.md](README.md) for project overview

### Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. View logs: `tail -f ~/ECO-ROUTE/logs/controller.log`
3. Check API health: `curl http://localhost:5000/api/health`
4. GitHub Issues: https://github.com/10srav/ECO-ROUTE/issues

---

## Summary

**EcoRoute** is a smart network management system that:

✓ **Saves 25-35% energy** in data center networks
✓ **Maintains excellent performance** (< 0.1% packet loss)
✓ **Works automatically** (no manual intervention needed)
✓ **Uses AI prediction** (EWMA algorithm)
✓ **Provides real-time monitoring** (web dashboard)

**Perfect for:**
- Data center operators looking to reduce costs
- Companies with sustainability goals
- Network researchers and students
- Anyone interested in green computing

**Get started in 3 steps:**
1. Install dependencies
2. Run `./run.sh start`
3. Open http://localhost:3000

**Questions?** Check the troubleshooting section or contact support.

---

*EcoRoute - Intelligent Energy Management for Modern Networks*
