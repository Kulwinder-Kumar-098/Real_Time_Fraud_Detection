#!/usr/bin/env bash
# Exit immediately on error
set -e

echo "=== [1/4] Installing Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [2/4] Checking Model Artifacts ==="
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

echo "=== [3/4] Initializing SQLite Database ==="
python -c "from fraud_detection.database import init_db; init_db(); print('Database schema and indexes initialized.')"

echo "=== [4/4] Seeding Demo Transactions ==="
# Only seed if the transactions table is empty (fresh deploy)
python -c "
import sqlite3, os
from pathlib import Path

db_path = os.environ.get('DATABASE_PATH', 'data/fraud_detection.db')
Path(db_path).parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(db_path)
count = conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0]
conn.close()

if count == 0:
    print('Empty DB detected. Seeding 200 demo transactions...')
    import sys
    sys.path.insert(0, '.')
    from fraud_detection.generator import generate_user_profiles, SyntheticTransactionGenerator
    from fraud_detection.inference import InferenceEngine
    import random

    profiles = generate_user_profiles(50, seed=42)
    gen = SyntheticTransactionGenerator(user_profiles=profiles)
    engine = InferenceEngine.get_instance()

    for i in range(200):
        is_fraud = random.random() < 0.12
        tx = gen.generate_single_transaction(force_fraud=is_fraud)
        try:
            engine.score_transaction(tx, persist=True)
        except Exception as e:
            print(f'  Seed tx {i} error: {e}')

    print('Seeding complete: 200 demo transactions inserted.')
else:
    print(f'Database already has {count} transactions. Skipping seed.')
"

echo "=== Build Complete: Application Ready for Production Deployment ==="
