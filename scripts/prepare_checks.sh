#!/usr/bin/env bash
# Preparation only: never trains, exports, or reads real holdout images.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/session_20260922
PY=.venv/bin/python
"$PY" -m pip check > results/session_20260922/pip_check.txt 2>&1
"$PY" -m pip freeze > results/session_20260922/dependencies.txt
"$PY" -m poultryrisk doctor > results/session_20260922/doctor.json 2> results/session_20260922/doctor_stderr.log
"$PY" -m unittest discover -s tests -v > results/session_20260922/unit_tests.log 2>&1
"$PY" scripts/statistical_smoke.py > results/session_20260922/statistical_smoke.log 2>&1
if [ ! -d data/synthetic ]; then
    "$PY" scripts/create_synthetic_dataset.py > results/session_20260922/synthetic_generation.log 2>&1
fi
if [ ! -f results/smoke_synthetic/data/manifest.csv ]; then
    "$PY" -m poultryrisk prepare --config configs/smoke.yaml > results/session_20260922/synthetic_prepare.log 2>&1
fi
"$PY" -m poultryrisk verify-data --config configs/smoke.yaml > results/session_20260922/synthetic_verify.log 2>&1
"$PY" -m poultryrisk feasibility --config configs/smoke.yaml > results/session_20260922/synthetic_feasibility.log 2>&1
echo 'Preparation checks passed. No training or real-data evaluation was run.'
