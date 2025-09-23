#!/bin/bash

# Comprehensive Quality Check Script for RAG Chatbot
# This script runs all quality checks and reports status without failing

set -e

echo "🔍 Comprehensive Code Quality Assessment"
echo "========================================"

# Function to run command and report status
run_assessment() {
    local name="$1"
    local cmd="$2"
    echo
    echo "📋 Assessing $name..."
    echo "---"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "✅ $name: PASSED"
        return 0
    else
        echo "⚠️  $name: NEEDS ATTENTION"
        return 1
    fi
}

# Track results
total_checks=0
passed_checks=0

# Run Black formatting check
run_assessment "Code formatting (Black)" "uv run black --check --diff ."
total_checks=$((total_checks + 1))
if [ $? -eq 0 ]; then passed_checks=$((passed_checks + 1)); fi

# Run isort import sorting check
run_assessment "Import sorting (isort)" "uv run isort --check-only --diff ."
total_checks=$((total_checks + 1))
if [ $? -eq 0 ]; then passed_checks=$((passed_checks + 1)); fi

# Run flake8 linting
run_assessment "Code linting (flake8)" "uv run flake8 backend/ main.py"
total_checks=$((total_checks + 1))
if [ $? -eq 0 ]; then passed_checks=$((passed_checks + 1)); fi

# Run mypy type checking
run_assessment "Type checking (mypy)" "uv run mypy backend/ main.py --no-error-summary"
total_checks=$((total_checks + 1))
if [ $? -eq 0 ]; then passed_checks=$((passed_checks + 1)); fi

# Run tests
run_assessment "Test suite (pytest)" "uv run pytest --tb=no -q"
total_checks=$((total_checks + 1))
if [ $? -eq 0 ]; then passed_checks=$((passed_checks + 1)); fi

echo
echo "📊 Quality Assessment Summary"
echo "============================="
echo "Passed: $passed_checks/$total_checks checks"

if [ $passed_checks -eq $total_checks ]; then
    echo "🎉 All quality checks passed! Code is ready for production."
    exit 0
else
    echo "⚠️  Some checks need attention. Use individual scripts to see details:"
    echo "   - ./scripts/format.sh    # Fix formatting"
    echo "   - ./scripts/lint.sh      # See linting details"
    echo "   - uv run pytest -v      # See test details"
    exit 0  # Don't fail, just report
fi