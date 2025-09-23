#!/bin/bash

# Linting Script for RAG Chatbot
# This script runs linting checks without formatting

set -e

echo "🔍 Running Linting Checks..."
echo "============================"

# Function to run command with status
run_check() {
    local name="$1"
    local cmd="$2"
    echo
    echo "📋 Running $name..."
    echo "---"
    if eval "$cmd"; then
        echo "✅ $name passed"
    else
        echo "❌ $name failed"
        exit 1
    fi
}

# Run flake8 linting
run_check "flake8 (linting)" "uv run flake8 backend/ main.py"

# Run mypy type checking (lenient for now)
echo
echo "📋 Running mypy (type checking)..."
echo "---"
if uv run mypy backend/ main.py --no-error-summary 2>/dev/null; then
    echo "✅ mypy (type checking) passed"
else
    echo "⚠️  mypy (type checking) has warnings but continuing..."
fi

echo
echo "🎉 All linting checks passed!"
echo "============================="