# EcoRoute Documentation Index

## 📚 Complete Documentation Guide

This repository contains comprehensive documentation for different audiences. Choose the guide that matches your needs:

---

## For Clients & Non-Technical Users

### [CLIENT_GUIDE.md](CLIENT_GUIDE.md) ⭐ **START HERE**

**Perfect for:** Business owners, managers, clients who want to understand the system

**Contains:**
- ✓ What is EcoRoute? (Simple explanation)
- ✓ How does it work? (Non-technical overview)
- ✓ Network topology explained with diagrams
- ✓ Complete setup instructions (step-by-step)
- ✓ How to use the dashboard
- ✓ Troubleshooting guide
- ✓ Frequently Asked Questions
- ✓ No technical jargon - explained like you're explaining to a friend

**Key Topics:**
- Understanding switches and network routing
- How energy is saved (25-35%)
- What the dashboard shows
- How to run the system
- Common problems and solutions

---

## For Technical Users & Operators

### [RUNNING_GUIDE.md](RUNNING_GUIDE.md) ⭐ **OPERATIONS GUIDE**

**Perfect for:** System administrators, DevOps engineers, technical operators

**Contains:**
- ✓ Quick start commands
- ✓ System architecture overview
- ✓ All operational commands
- ✓ API endpoint reference
- ✓ Monitoring and debugging
- ✓ Log file locations
- ✓ Troubleshooting procedures
- ✓ Quick reference commands

**Key Topics:**
- Starting/stopping services
- Monitoring system health
- Debugging issues
- API usage
- Log analysis

---

## For Developers & Contributors

### [CLAUDE.md](CLAUDE.md) ⭐ **DEVELOPER GUIDE**

**Perfect for:** Developers working on the codebase

**Contains:**
- ✓ Core component architecture
- ✓ Code structure and organization
- ✓ Key algorithms and implementation details
- ✓ Development workflow
- ✓ Testing procedures
- ✓ Code quality tools
- ✓ Contributing guidelines

**Key Topics:**
- Component interaction
- EWMA prediction algorithm
- Energy-aware routing logic
- Make-before-break pattern
- Testing strategy
- Code modification guidelines

---

## Project Overview

### [README.md](README.md) ⭐ **PROJECT README**

**Perfect for:** First-time visitors to the repository

**Contains:**
- ✓ Project introduction
- ✓ Key features
- ✓ Quick start guide
- ✓ Architecture overview
- ✓ Configuration details
- ✓ API endpoints
- ✓ Contributing information

---

## Quick Reference

### Which guide should I read?

| Your Role | Start With | Then Read |
|-----------|------------|-----------|
| **Client/Manager** | [CLIENT_GUIDE.md](CLIENT_GUIDE.md) | [README.md](README.md) |
| **System Admin** | [RUNNING_GUIDE.md](RUNNING_GUIDE.md) | [CLIENT_GUIDE.md](CLIENT_GUIDE.md) |
| **Developer** | [CLAUDE.md](CLAUDE.md) | [RUNNING_GUIDE.md](RUNNING_GUIDE.md) |
| **First Time User** | [README.md](README.md) | [CLIENT_GUIDE.md](CLIENT_GUIDE.md) |

---

## Documentation Summary

### What You'll Learn

**CLIENT_GUIDE.md:**
- 🎯 Business value and energy savings
- 🌐 How the network works (simple explanation)
- 🚀 Complete setup from scratch
- 📊 Dashboard interpretation
- 🔧 Problem solving

**RUNNING_GUIDE.md:**
- ⚡ Quick start commands
- 🛠️ System operations
- 📡 API usage
- 🐛 Debugging techniques
- 📝 Log analysis

**CLAUDE.md:**
- 💻 Code architecture
- 🔬 Algorithm details
- 🧪 Testing procedures
- 📦 Code organization
- 🔄 Development workflow

---

## Additional Resources

### Scripts & Tools

- **push_to_github.sh** - Helper script to push changes to GitHub
- **run.sh** - Quick start/stop script for the system
- **config.yaml** - System configuration file

### Logs & Monitoring

- **logs/controller.log** - Controller activity and events
- **logs/dashboard.log** - Dashboard API logs
- **logs/frontend.log** - React frontend logs
- **logs/metrics.csv** - Exported performance metrics

---

## Getting Help

### For Questions About:

**Setup & Installation**
→ Read: [CLIENT_GUIDE.md - Step 1-4](CLIENT_GUIDE.md#step-1-install-system-dependencies)

**Running the System**
→ Read: [RUNNING_GUIDE.md - Start Everything](RUNNING_GUIDE.md#start-everything)

**Understanding Energy Savings**
→ Read: [CLIENT_GUIDE.md - How Does It Work](CLIENT_GUIDE.md#how-does-it-work)

**API Integration**
→ Read: [RUNNING_GUIDE.md - API Endpoints](RUNNING_GUIDE.md#api-endpoints)

**Code Modifications**
→ Read: [CLAUDE.md - Working with the Codebase](CLAUDE.md#working-with-the-codebase)

**Troubleshooting**
→ Read: [CLIENT_GUIDE.md - Troubleshooting](CLIENT_GUIDE.md#troubleshooting)

---

## Document Version Control

| Document | Last Updated | Status |
|----------|--------------|--------|
| CLIENT_GUIDE.md | 2026-01-31 | ✅ Complete |
| RUNNING_GUIDE.md | 2026-01-31 | ✅ Complete |
| CLAUDE.md | 2026-01-31 | ✅ Complete |
| README.md | Original | ✅ Complete |
| DOCUMENTATION_INDEX.md | 2026-01-31 | ✅ Current |

---

## Feedback & Contributions

Found an issue or have suggestions for the documentation?

1. **GitHub Issues:** https://github.com/10srav/ECO-ROUTE/issues
2. **Pull Requests:** Submit improvements via PR
3. **Email:** Contact the development team

---

## Quick Start Based on Your Goal

### 🎯 Goal: Just want to see it running
**Read:** [CLIENT_GUIDE.md - Step 5](CLIENT_GUIDE.md#step-5-start-the-system)
**Time:** 5 minutes

### 🎯 Goal: Understand how it works
**Read:** [CLIENT_GUIDE.md - How Does It Work](CLIENT_GUIDE.md#how-does-it-work)
**Time:** 10 minutes

### 🎯 Goal: Setup from scratch
**Read:** [CLIENT_GUIDE.md - Complete Setup](CLIENT_GUIDE.md#complete-setup-guide-for-clients)
**Time:** 30 minutes

### 🎯 Goal: Integrate with my system
**Read:** [RUNNING_GUIDE.md](RUNNING_GUIDE.md) + [CLAUDE.md](CLAUDE.md)
**Time:** 1-2 hours

### 🎯 Goal: Modify the code
**Read:** [CLAUDE.md](CLAUDE.md)
**Time:** 2-4 hours

---

## System Requirements Reminder

- **OS:** Ubuntu Linux 20.04+
- **Python:** 3.8+ (tested on 3.12)
- **RAM:** 4+ GB
- **Disk:** 10+ GB
- **Network:** Mininet support
- **Access:** Root/sudo privileges

---

**Need help?** Start with the guide that matches your role above! 🚀

---

*Last Updated: January 31, 2026*
*EcoRoute Project Documentation v1.0*
