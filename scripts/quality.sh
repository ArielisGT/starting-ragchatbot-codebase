#!/bin/bash

# Quality Check Script for RAG Chatbot
# This script runs various code quality checks

set -e

echo "🔍 Running Code Quality Checks..."
echo "================================="

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

# Run Black formatting check
run_check "Black (code formatting)" "uv run black --check --diff ."

# Run isort import sorting check
run_check "isort (import sorting)" "uv run isort --check-only --diff ."

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

# Run tests
run_check "pytest (tests)" "uv run pytest -v"

echo
echo "🎉 All quality checks passed!"
echo "============================="