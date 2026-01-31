#!/bin/bash
# EcoRoute - Push to GitHub Helper Script

echo "============================================"
echo "EcoRoute - Push Changes to GitHub"
echo "============================================"
echo ""

# Check if there are changes to push
cd "$(dirname "$0")"

# Show current status
echo "Current commit:"
git log --oneline -1
echo ""

echo "Files changed:"
git show --stat --oneline HEAD | tail -n +2
echo ""

echo "============================================"
echo "Ready to push to: https://github.com/10srav/ECO-ROUTE.git"
echo ""
echo "You will be prompted for:"
echo "  Username: Your GitHub username"
echo "  Password: Your Personal Access Token (NOT your GitHub password)"
echo ""
echo "Don't have a token? Create one at:"
echo "  https://github.com/settings/tokens/new"
echo "  - Give it 'repo' permissions"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Push to GitHub
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ Successfully pushed to GitHub!"
    echo "============================================"
    echo ""
    echo "View your changes at:"
    echo "  https://github.com/10srav/ECO-ROUTE"
    echo ""
else
    echo ""
    echo "============================================"
    echo "✗ Push failed!"
    echo "============================================"
    echo ""
    echo "Common issues:"
    echo "  1. Wrong username or token"
    echo "  2. Token doesn't have 'repo' permissions"
    echo "  3. Network connection issues"
    echo ""
    echo "Try again or use SSH authentication instead."
fi
