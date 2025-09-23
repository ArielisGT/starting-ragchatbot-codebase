#!/bin/bash

# Code Formatting Script for RAG Chatbot
# This script automatically formats code using Black and isort

set -e

echo "🎨 Formatting Code..."
echo "===================="

echo
echo "📋 Running Black (code formatting)..."
uv run black .

echo
echo "📋 Running isort (import sorting)..."
uv run isort .

echo
echo "✅ Code formatting complete!"
echo "============================"