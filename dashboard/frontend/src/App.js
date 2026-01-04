import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Legend
} from 'recharts';

const API_BASE = process.env.REACT_APP_API_URL || '';

function App() {
  const [stats, setStats] = useState(null);
  const [topology, setTopology] = useState(null);
  const [energyHistory, setEnergyHistory] = useState([]);
  const [events, setEvents] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [connected, setConnected] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(null);

  // Fetch data from API
  const fetchData = useCallback(async () => {
    try {
      const [statsRes, topoRes, historyRes, eventsRes, predsRes] = await Promise.all([
        axios.get(`${API_BASE}/api/stats`),
        axios.get(`${API_BASE}/api/topology`),
        axios.get(`${API_BASE}/api/energy/history`),
        axios.get(`${API_BASE}/api/events`),
        axios.get(`${API_BASE}/api/predictions`)
      ]);

      setStats(statsRes.data);
      setTopology(topoRes.data);
      setEnergyHistory(historyRes.data.history || []);
      setEvents(eventsRes.data.events || []);
      setPredictions(predsRes.data.predictions || []);
      setConnected(true);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch data:', error);
      setConnected(false);
    }
  }, []);

  // Poll for updates
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Format timestamp
  const formatTime = (timestamp) => {
    if (!timestamp) return '--';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
  };

  // Get color for utilization
  const getUtilColor = (util) => {
    if (util < 40) return 'green';
    if (util < 70) return 'yellow';
    return 'red';
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-title">
          <span role="img" aria-label="leaf">🌿</span>
          <h1>EcoRoute Dashboard</h1>
        </div>
        <div className="header-status">
          <span className={`status-dot ${connected ? '' : 'disconnected'}`}
                style={{ backgroundColor: connected ? '#00ff88' : '#ff4757' }}></span>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
          {lastUpdate && (
            <span style={{ marginLeft: '1rem' }}>
              Last update: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
      </header>

      {/* Dashboard Grid */}
      <main className="dashboard">
        {/* Metric Cards */}
        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">Energy Savings</span>
            <span className="metric-card-icon">⚡</span>
          </div>
          <div className="metric-card-value green">
            {stats?.energy?.energy_savings_percent?.toFixed(1) || '0.0'}%
          </div>
          <div className="metric-card-subtitle">
            vs ECMP baseline
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">Active Ports</span>
            <span className="metric-card-icon">🔌</span>
          </div>
          <div className="metric-card-value blue">
            {stats?.energy?.active_ports || 0}
          </div>
          <div className="metric-card-subtitle">
            of {stats?.energy?.total_ports || 0} total
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">Sleeping Links</span>
            <span className="metric-card-icon">😴</span>
          </div>
          <div className="metric-card-value yellow">
            {stats?.energy?.sleeping_ports || 0}
          </div>
          <div className="metric-card-subtitle">
            links in sleep mode
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-header">
            <span className="metric-card-title">Power Saved</span>
            <span className="metric-card-icon">💡</span>
          </div>
          <div className="metric-card-value green">
            {stats?.energy?.power_saved_watts?.toFixed(0) || '0'}W
          </div>
          <div className="metric-card-subtitle">
            current savings
          </div>
        </div>

        {/* Energy Savings Chart */}
        <div className="chart-container large">
          <h3 className="chart-title">
            <span role="img" aria-label="chart">📈</span>
            Energy Savings Over Time
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={energyHistory.slice(-30)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatTime}
                stroke="#a0a0a0"
                fontSize={12}
              />
              <YAxis
                stroke="#a0a0a0"
                fontSize={12}
                domain={[0, 50]}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#16213e',
                  border: '1px solid #2a2a4a',
                  borderRadius: '8px'
                }}
                labelFormatter={formatTime}
              />
              <Area
                type="monotone"
                dataKey="savings"
                stroke="#00ff88"
                fill="url(#greenGradient)"
                name="Energy Savings %"
              />
              <defs>
                <linearGradient id="greenGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00ff88" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#00ff88" stopOpacity={0}/>
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Power Consumption Chart */}
        <div className="chart-container large">
          <h3 className="chart-title">
            <span role="img" aria-label="power">⚡</span>
            Power Consumption
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={[
              {
                name: 'Current',
                power: stats?.energy?.total_power_watts || 0
              },
              {
                name: 'Baseline',
                power: stats?.energy?.baseline_power_watts || 0
              }
            ]}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis dataKey="name" stroke="#a0a0a0" fontSize={12} />
              <YAxis
                stroke="#a0a0a0"
                fontSize={12}
                tickFormatter={(v) => `${v}W`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#16213e',
                  border: '1px solid #2a2a4a',
                  borderRadius: '8px'
                }}
                formatter={(value) => [`${value.toFixed(1)}W`, 'Power']}
              />
              <Bar dataKey="power" fill="#4dabf7" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Topology View */}
        <div className="topology-container">
          <h3 className="chart-title">
            <span role="img" aria-label="network">🌐</span>
            Network Topology
          </h3>
          <TopologyView topology={topology} />
        </div>

        {/* QoS Metrics */}
        <div className="chart-container">
          <h3 className="chart-title">
            <span role="img" aria-label="gauge">📊</span>
            QoS Metrics
          </h3>
          <div className="qos-indicators">
            <div className="qos-indicator">
              <span className="qos-label">Packet Loss</span>
              <span className="qos-value" style={{
                color: (stats?.qos?.max_packet_loss || 0) < 0.1 ? '#00ff88' : '#ff4757'
              }}>
                {((stats?.qos?.max_packet_loss || 0) * 100).toFixed(2)}%
              </span>
            </div>
            <div className="qos-indicator">
              <span className="qos-label">Max Latency</span>
              <span className="qos-value" style={{
                color: (stats?.qos?.max_latency_ms || 0) < 5 ? '#00ff88' : '#ffd43b'
              }}>
                {(stats?.qos?.max_latency_ms || 0).toFixed(1)}ms
              </span>
            </div>
            <div className="qos-indicator">
              <span className="qos-label">Throughput</span>
              <span className="qos-value green">
                {((stats?.qos?.throughput_ratio || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="qos-indicator">
              <span className="qos-label">QoS Violations</span>
              <span className="qos-value" style={{
                color: (stats?.qos?.qos_violations || 0) === 0 ? '#00ff88' : '#ff4757'
              }}>
                {stats?.qos?.qos_violations || 0}
              </span>
            </div>
          </div>
        </div>

        {/* Events Panel */}
        <div className="events-panel">
          <h3 className="chart-title">
            <span role="img" aria-label="events">📋</span>
            Recent Events
          </h3>
          <ul className="events-list">
            {events.slice(-10).reverse().map((event, idx) => (
              <li key={idx} className="event-item">
                <span className={`event-type ${event.type.includes('sleep') ? 'sleep' : event.type.includes('wake') ? 'wake' : 'reroute'}`}>
                  {event.type.replace('_', ' ')}
                </span>
                <span>{event.details}</span>
                <span className="event-time">{formatTime(event.timestamp)}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* EWMA Predictions */}
        <div className="chart-container large">
          <h3 className="chart-title">
            <span role="img" aria-label="predict">🔮</span>
            EWMA Traffic Predictions
          </h3>
          <div style={{ maxHeight: '250px', overflow: 'auto' }}>
            <table className="predictions-table">
              <thead>
                <tr>
                  <th>Link</th>
                  <th>Current Load</th>
                  <th>Predicted</th>
                  <th>Confidence</th>
                  <th>Trend</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 10).map((pred, idx) => (
                  <tr key={idx}>
                    <td>{pred.link}</td>
                    <td>
                      <div className="progress-bar" style={{ width: '80px' }}>
                        <div
                          className={`progress-fill ${getUtilColor(pred.current_load)}`}
                          style={{ width: `${pred.current_load}%` }}
                        />
                      </div>
                      <span style={{ marginLeft: '0.5rem', fontSize: '0.75rem' }}>
                        {pred.current_load.toFixed(1)}%
                      </span>
                    </td>
                    <td>{pred.predicted_load.toFixed(1)}%</td>
                    <td>{(pred.confidence * 100).toFixed(0)}%</td>
                    <td>
                      <span className={`trend-indicator trend-${pred.trend === 'increasing' ? 'up' : pred.trend === 'decreasing' ? 'down' : 'stable'}`}>
                        {pred.trend === 'increasing' ? '↑' : pred.trend === 'decreasing' ? '↓' : '→'}
                        {pred.trend}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ECMP Comparison */}
        <div className="chart-container large">
          <h3 className="chart-title">
            <span role="img" aria-label="compare">⚖️</span>
            ECMP Baseline Comparison
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <h4 style={{ fontSize: '0.875rem', color: '#a0a0a0', marginBottom: '0.5rem' }}>
                Energy Consumption
              </h4>
              <ResponsiveContainer width="100%" height={150}>
                <BarChart data={[
                  { name: 'Baseline', value: stats?.ecmp_comparison?.baseline_energy_watts || 0, fill: '#ff4757' },
                  { name: 'EcoRoute', value: stats?.ecmp_comparison?.current_energy_watts || 0, fill: '#00ff88' }
                ]} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                  <XAxis type="number" stroke="#a0a0a0" fontSize={12} />
                  <YAxis dataKey="name" type="category" stroke="#a0a0a0" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#16213e',
                      border: '1px solid #2a2a4a',
                      borderRadius: '8px'
                    }}
                    formatter={(value) => [`${value.toFixed(1)}W`, 'Power']}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h4 style={{ fontSize: '0.875rem', color: '#a0a0a0', marginBottom: '0.5rem' }}>
                Key Metrics
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1rem' }}>
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#a0a0a0' }}>Energy Savings</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#00ff88' }}>
                    {stats?.ecmp_comparison?.energy_savings_percent?.toFixed(1) || '0'}%
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#a0a0a0' }}>Power Saved</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#4dabf7' }}>
                    {stats?.ecmp_comparison?.energy_savings_watts?.toFixed(1) || '0'}W
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: '#a0a0a0' }}>Active Ports Reduction</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#ffd43b' }}>
                    {stats?.ecmp_comparison?.active_ports_reduction_percent?.toFixed(1) || '0'}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// Topology Visualization Component
function TopologyView({ topology }) {
  if (!topology || !topology.nodes) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '300px',
        color: '#a0a0a0'
      }}>
        Loading topology...
      </div>
    );
  }

  const switches = topology.nodes.filter(n => n.type !== 'host');
  const edges = topology.edges.filter(e =>
    !e.target.startsWith('h') && !e.source.startsWith('h')
  );

  // Simple force-directed layout simulation
  const width = 500;
  const height = 350;

  // Position nodes by type
  const coreY = 50;
  const aggY = 150;
  const edgeY = 250;

  const positionedNodes = switches.map((node, idx) => {
    let x, y;
    if (node.type === 'core') {
      const coreNodes = switches.filter(n => n.type === 'core');
      const coreIdx = coreNodes.indexOf(node);
      x = (width / (coreNodes.length + 1)) * (coreIdx + 1);
      y = coreY;
    } else if (node.type === 'aggregation') {
      const aggNodes = switches.filter(n => n.type === 'aggregation');
      const aggIdx = aggNodes.indexOf(node);
      x = (width / (aggNodes.length + 1)) * (aggIdx + 1);
      y = aggY;
    } else {
      const edgeNodes = switches.filter(n => n.type === 'edge');
      const edgeIdx = edgeNodes.indexOf(node);
      x = (width / (edgeNodes.length + 1)) * (edgeIdx + 1);
      y = edgeY;
    }
    return { ...node, x, y };
  });

  const nodeMap = {};
  positionedNodes.forEach(n => { nodeMap[n.id] = n; });

  return (
    <svg className="topology-svg" viewBox={`0 0 ${width} ${height}`}>
      {/* Draw edges */}
      {edges.map((edge, idx) => {
        const source = nodeMap[edge.source];
        const target = nodeMap[edge.target];
        if (!source || !target) return null;

        return (
          <line
            key={idx}
            x1={source.x}
            y1={source.y}
            x2={target.x}
            y2={target.y}
            className={`topology-link ${edge.sleeping ? 'sleeping' : 'active'}`}
            stroke={edge.sleeping ? '#ff4757' : '#00ff88'}
            strokeOpacity={0.6}
          />
        );
      })}

      {/* Draw nodes */}
      {positionedNodes.map((node, idx) => (
        <g key={idx} className="topology-node" transform={`translate(${node.x}, ${node.y})`}>
          {node.type === 'core' && (
            <rect x="-15" y="-10" width="30" height="20" rx="3" fill="#4dabf7" />
          )}
          {node.type === 'aggregation' && (
            <rect x="-12" y="-8" width="24" height="16" rx="3" fill="#ffd43b" />
          )}
          {node.type === 'edge' && (
            <rect x="-10" y="-6" width="20" height="12" rx="2" fill="#00ff88" />
          )}
          <text
            textAnchor="middle"
            dy="25"
            fontSize="8"
            fill="#a0a0a0"
          >
            {node.id}
          </text>
        </g>
      ))}

      {/* Legend */}
      <g transform="translate(10, 310)">
        <rect x="0" y="0" width="10" height="10" fill="#4dabf7" />
        <text x="15" y="9" fontSize="9" fill="#a0a0a0">Core</text>

        <rect x="50" y="0" width="10" height="10" fill="#ffd43b" />
        <text x="65" y="9" fontSize="9" fill="#a0a0a0">Aggregation</text>

        <rect x="130" y="0" width="10" height="10" fill="#00ff88" />
        <text x="145" y="9" fontSize="9" fill="#a0a0a0">Edge</text>

        <line x1="200" y1="5" x2="220" y2="5" stroke="#00ff88" strokeWidth="2" />
        <text x="225" y="9" fontSize="9" fill="#a0a0a0">Active</text>

        <line x1="270" y1="5" x2="290" y2="5" stroke="#ff4757" strokeWidth="2" strokeDasharray="3,3" />
        <text x="295" y="9" fontSize="9" fill="#a0a0a0">Sleeping</text>
      </g>
    </svg>
  );
}

export default App;
