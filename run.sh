#!/bin/bash
# AI Candidate Ranking System - Quick Start Wrapper

set -e

echo "=========================================================="
echo "🎯 Redrob Hackathon: AI Candidate Ranker Setup & Run"
echo "=========================================================="

echo "[1/3] Checking environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "[2/3] Installing dependencies..."
python3 -m pip install -r requirements.txt -q
echo "✅ Dependencies installed."

echo "[3/3] Running the ranking pipeline..."
# Optimize thread usage to prevent macOS fork issues
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Run the pipeline with verbose output so judges can see the magic happening
python3 rank.py --candidates data/raw/sample_candidates.json --out data/output/submission.csv --verbose

echo ""
echo "✅ Done! Top candidates have been saved to data/output/submission.csv"
echo ""
echo "Want to see the interactive dashboard? Run: streamlit run app.py"
echo "=========================================================="
