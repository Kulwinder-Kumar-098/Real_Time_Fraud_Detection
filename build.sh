#!/usr/bin/env bash
# Exit immediately on error
set -e

echo "=== [1/3] Installing Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [2/3] Checking Model Artifacts ==="
# Train model only if artifacts are not already present (prevents redundant build retraining)
python -c "
from pathlib import Path
import os
import sys

art_path = Path('models/model_artifacts.joblib')
meta_path = Path('models/model_metadata.json')

if not (art_path.exists() and meta_path.exists()):
    print('Artifacts missing. Running initial offline model training...')
    exit_code = os.system('python -m fraud_detection.train')
    if exit_code != 0:
        sys.exit(1)
else:
    print(f'Model artifacts verified ({art_path.stat().st_size} bytes). Skipping retraining.')
"

echo "=== [3/3] Initializing SQLite Database ==="
python -c "from fraud_detection.database import init_db; init_db(); print('Database schema and indexes initialized.')"

echo "=== Build Complete: Application Ready for Production Deployment ==="
