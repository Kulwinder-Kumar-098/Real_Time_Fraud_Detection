import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    home_location TEXT NOT NULL,
    primary_device TEXT NOT NULL,
    default_payment_method TEXT NOT NULL,
    avg_amount REAL NOT NULL,
    std_amount REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    amount REAL NOT NULL,
    timestamp TEXT NOT NULL,
    merchant TEXT NOT NULL,
    merchant_category TEXT NOT NULL,
    location TEXT NOT NULL,
    device TEXT NOT NULL,
    payment_method TEXT NOT NULL,
    channel TEXT NOT NULL,
    ground_truth_label INTEGER DEFAULT 0,
    fraud_scenario TEXT DEFAULT 'none',
    risk_score REAL DEFAULT 0.0,
    is_fraud_predicted INTEGER DEFAULT 0,
    decision TEXT DEFAULT 'APPROVE',
    risk_tier TEXT DEFAULT 'LOW',
    risk_reasons TEXT DEFAULT '[]',
    latency_ms REAL DEFAULT 0.0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transactions_txid ON transactions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_transactions_userid ON transactions (user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions (timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_predicted ON transactions (is_fraud_predicted);
CREATE INDEX IF NOT EXISTS idx_transactions_decision ON transactions (decision);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    threshold REAL NOT NULL,
    roc_auc REAL,
    pr_auc REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    total_cost REAL,
    confusion_matrix TEXT,
    evaluated_at TEXT NOT NULL
);
"""


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Provides a thread-safe sqlite3 connection with dict-like row access."""
    path = db_path or config.DATABASE_PATH
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables and indexes safely."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executescript(SCHEMA_SQL)
    finally:
        conn.close()


def upsert_user(user_data: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Insert or update a user profile."""
    conn = get_connection(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO users (user_id, home_location, primary_device, default_payment_method,
                                  avg_amount, std_amount, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    home_location=excluded.home_location,
                    primary_device=excluded.primary_device,
                    default_payment_method=excluded.default_payment_method,
                    avg_amount=excluded.avg_amount,
                    std_amount=excluded.std_amount;
                """,
                (
                    user_data["user_id"],
                    user_data["home_location"],
                    user_data["primary_device"],
                    user_data["default_payment_method"],
                    float(user_data.get("avg_amount", 50.0)),
                    float(user_data.get("std_amount", 20.0)),
                    user_data.get("created_at", now_iso),
                ),
            )
    finally:
        conn.close()


def upsert_users_batch(users: List[Dict[str, Any]], db_path: Optional[str] = None) -> None:
    """Insert or update multiple user profiles in a single transaction."""
    if not users:
        return
    conn = get_connection(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            conn.executemany(
                """
                INSERT INTO users (user_id, home_location, primary_device, default_payment_method,
                                  avg_amount, std_amount, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    home_location=excluded.home_location,
                    primary_device=excluded.primary_device,
                    default_payment_method=excluded.default_payment_method,
                    avg_amount=excluded.avg_amount,
                    std_amount=excluded.std_amount;
                """,
                [
                    (
                        u["user_id"],
                        u["home_location"],
                        u["primary_device"],
                        u["default_payment_method"],
                        float(u.get("avg_amount", 50.0)),
                        float(u.get("std_amount", 20.0)),
                        u.get("created_at", now_iso),
                    )
                    for u in users
                ],
            )
    finally:
        conn.close()


def get_user(user_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch user profile by user_id."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_users(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch all user profiles."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM users").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def insert_transaction(tx: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Insert a scored transaction into the database."""
    conn = get_connection(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    reasons_json = (
        tx["risk_reasons"]
        if isinstance(tx.get("risk_reasons"), str)
        else json.dumps(tx.get("risk_reasons", []))
    )

    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO transactions (
                    transaction_id, user_id, amount, timestamp, merchant,
                    merchant_category, location, device, payment_method, channel,
                    ground_truth_label, fraud_scenario, risk_score, is_fraud_predicted,
                    decision, risk_tier, risk_reasons, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(tx["transaction_id"]),
                    str(tx["user_id"]),
                    float(tx["amount"]),
                    str(tx["timestamp"]),
                    str(tx["merchant"]),
                    str(tx.get("merchant_category", "General")),
                    str(tx["location"]),
                    str(tx["device"]),
                    str(tx["payment_method"]),
                    str(tx.get("channel", "Online")),
                    int(tx.get("ground_truth_label", 0)),
                    str(tx.get("fraud_scenario", "none")),
                    float(tx.get("risk_score", 0.0)),
                    int(tx.get("is_fraud_predicted", 0)),
                    str(tx.get("decision", "APPROVE")),
                    str(tx.get("risk_tier", "LOW")),
                    reasons_json,
                    float(tx.get("latency_ms", 0.0)),
                    tx.get("created_at", now_iso),
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_user_recent_activity(
    user_id: str,
    reference_timestamp: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes rolling transaction features for a user:
    - tx count in last 1 hour
    - tx count in last 24 hours
    - sum amount in last 1 hour
    - last seen location
    - last seen device
    """
    conn = get_connection(db_path)
    try:
        # If reference timestamp is provided, calculate relative to it; else now
        ref_time = reference_timestamp or datetime.now(timezone.utc).isoformat()

        # Last transaction
        last_tx = conn.execute(
            """
            SELECT location, device, timestamp, amount
            FROM transactions
            WHERE user_id = ? AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (user_id, ref_time),
        ).fetchone()

        # Transactions in last 1 hour and 24 hours (using SQLite datetime functions or string comparison)
        stats = conn.execute(
            """
            SELECT
                COUNT(CASE WHEN timestamp >= datetime(?, '-1 hour') THEN 1 END) as tx_count_1h,
                COUNT(CASE WHEN timestamp >= datetime(?, '-24 hours') THEN 1 END) as tx_count_24h,
                COALESCE(SUM(CASE WHEN timestamp >= datetime(?, '-1 hour') THEN amount ELSE 0 END), 0.0) as amount_sum_1h,
                COALESCE(AVG(amount), 0.0) as hist_avg_amount
            FROM transactions
            WHERE user_id = ? AND timestamp <= ?
            """,
            (ref_time, ref_time, ref_time, user_id, ref_time),
        ).fetchone()

        return {
            "tx_count_1h": stats["tx_count_1h"] if stats else 0,
            "tx_count_24h": stats["tx_count_24h"] if stats else 0,
            "amount_sum_1h": stats["amount_sum_1h"] if stats else 0.0,
            "hist_avg_amount": stats["hist_avg_amount"] if stats else 0.0,
            "last_location": last_tx["location"] if last_tx else None,
            "last_device": last_tx["device"] if last_tx else None,
            "last_timestamp": last_tx["timestamp"] if last_tx else None,
        }
    finally:
        conn.close()


def get_recent_transactions(
    limit: int = 50,
    offset: int = 0,
    decision_filter: Optional[str] = None,
    risk_tier: Optional[str] = None,
    search: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Retrieve transactions with pagination, search, and filtering."""
    conn = get_connection(db_path)
    try:
        conditions = []
        params: List[Any] = []

        if decision_filter and decision_filter != "ALL":
            conditions.append("decision = ?")
            params.append(decision_filter)

        if risk_tier and risk_tier != "ALL":
            conditions.append("risk_tier = ?")
            params.append(risk_tier)

        if search:
            search_pattern = f"%{search.strip()}%"
            conditions.append(
                "(transaction_id LIKE ? OR user_id LIKE ? OR merchant LIKE ? OR location LIKE ?)"
            )
            params.extend([search_pattern, search_pattern, search_pattern, search_pattern])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Total count query
        count_query = f"SELECT COUNT(*) FROM transactions {where_clause}"
        total_count = conn.execute(count_query, params).fetchone()[0]

        # Paginated rows query
        query = f"""
            SELECT * FROM transactions
            {where_clause}
            ORDER BY timestamp DESC, id DESC
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, params + [limit, offset]).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("risk_reasons"), str):
                try:
                    d["risk_reasons"] = json.loads(d["risk_reasons"])
                except Exception:
                    d["risk_reasons"] = []
            results.append(d)

        return results, total_count
    finally:
        conn.close()


def get_transaction_by_id(tx_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve full transaction details by transaction_id."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?", (tx_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if isinstance(d.get("risk_reasons"), str):
            try:
                d["risk_reasons"] = json.loads(d["risk_reasons"])
            except Exception:
                d["risk_reasons"] = []
        return d
    finally:
        conn.close()


def get_dashboard_kpis(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculates live KPI metrics for dashboard."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) as total_count,
                COALESCE(SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END), 0) as fraud_predicted_count,
                COALESCE(SUM(CASE WHEN ground_truth_label = 1 THEN 1 ELSE 0 END), 0) as ground_truth_fraud_count,
                COALESCE(SUM(CASE WHEN decision = 'DECLINE' THEN 1 ELSE 0 END), 0) as decline_count,
                COALESCE(SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END), 0) as review_count,
                COALESCE(SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END), 0) as approve_count,
                COALESCE(AVG(amount), 0.0) as avg_amount,
                COALESCE(SUM(amount), 0.0) as total_volume,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms,
                COALESCE(MAX(latency_ms), 0.0) as max_latency_ms
            FROM transactions
            """
        ).fetchone()

        total = row["total_count"] if row else 0
        fraud_pred = row["fraud_predicted_count"] if row else 0
        fraud_actual = row["ground_truth_fraud_count"] if row else 0

        pred_fraud_rate = (fraud_pred / total * 100.0) if total > 0 else 0.0
        actual_fraud_rate = (fraud_actual / total * 100.0) if total > 0 else 0.0

        # Confusion statistics if ground_truth is populated
        confusion = conn.execute(
            """
            SELECT
                SUM(CASE WHEN ground_truth_label = 1 AND is_fraud_predicted = 1 THEN 1 ELSE 0 END) as tp,
                SUM(CASE WHEN ground_truth_label = 0 AND is_fraud_predicted = 1 THEN 1 ELSE 0 END) as fp,
                SUM(CASE WHEN ground_truth_label = 0 AND is_fraud_predicted = 0 THEN 1 ELSE 0 END) as tn,
                SUM(CASE WHEN ground_truth_label = 1 AND is_fraud_predicted = 0 THEN 1 ELSE 0 END) as fn
            FROM transactions
            WHERE ground_truth_label IS NOT NULL
            """
        ).fetchone()

        tp = confusion["tp"] or 0
        fp = confusion["fp"] or 0
        tn = confusion["tn"] or 0
        fn = confusion["fn"] or 0

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "total_transactions": total,
            "fraud_predicted_count": fraud_pred,
            "predicted_fraud_rate": round(pred_fraud_rate, 2),
            "actual_fraud_rate": round(actual_fraud_rate, 2),
            "decline_count": row["decline_count"],
            "review_count": row["review_count"],
            "approve_count": row["approve_count"],
            "avg_amount": round(row["avg_amount"], 2),
            "total_volume": round(row["total_volume"], 2),
            "avg_latency_ms": round(row["avg_latency_ms"], 2),
            "max_latency_ms": round(row["max_latency_ms"], 2),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "live_metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
        }
    finally:
        conn.close()


def get_volume_timeline(limit_points: int = 24, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns transaction counts and fraud counts grouped by recent hour or bucket."""
    conn = get_connection(db_path)
    try:
        # Group by hour using SQLite strftime
        rows = conn.execute(
            """
            SELECT
                strftime('%Y-%m-%d %H:00', timestamp) as time_bucket,
                COUNT(*) as total_count,
                SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) as fraud_count,
                SUM(CASE WHEN ground_truth_label = 1 THEN 1 ELSE 0 END) as actual_fraud_count,
                AVG(amount) as avg_amount
            FROM transactions
            GROUP BY time_bucket
            ORDER BY time_bucket DESC
            LIMIT ?
            """,
            (limit_points,),
        ).fetchall()

        timeline = [dict(r) for r in reversed(rows)]
        return timeline
    finally:
        conn.close()


def get_category_risk_breakdown(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns fraud breakdown across merchant categories."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                merchant_category,
                COUNT(*) as total_count,
                SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) as fraud_count,
                ROUND(AVG(amount), 2) as avg_amount,
                ROUND(CAST(SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) AS REAL) * 100.0 / COUNT(*), 2) as fraud_rate
            FROM transactions
            GROUP BY merchant_category
            ORDER BY total_count DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_risk_score_distribution(db_path: Optional[str] = None) -> Dict[str, int]:
    """Returns count of transactions across risk score intervals."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN risk_score < 0.20 THEN 1 ELSE 0 END) as band_0_20,
                SUM(CASE WHEN risk_score >= 0.20 AND risk_score < 0.40 THEN 1 ELSE 0 END) as band_20_40,
                SUM(CASE WHEN risk_score >= 0.40 AND risk_score < 0.60 THEN 1 ELSE 0 END) as band_40_60,
                SUM(CASE WHEN risk_score >= 0.60 AND risk_score < 0.80 THEN 1 ELSE 0 END) as band_60_80,
                SUM(CASE WHEN risk_score >= 0.80 THEN 1 ELSE 0 END) as band_80_100
            FROM transactions
            """
        ).fetchone()
        return {
            "0.0 - 0.2": row["band_0_20"] or 0,
            "0.2 - 0.4": row["band_20_40"] or 0,
            "0.4 - 0.6": row["band_40_60"] or 0,
            "0.6 - 0.8": row["band_60_80"] or 0,
            "0.8 - 1.0": row["band_80_100"] or 0,
        }
    finally:
        conn.close()


def record_model_metrics(
    model_name: str,
    threshold: float,
    metrics: Dict[str, Any],
    confusion: Dict[str, int],
    db_path: Optional[str] = None,
) -> None:
    """Store model training and evaluation metrics into model_metrics table."""
    conn = get_connection(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO model_metrics (
                    model_name, threshold, roc_auc, pr_auc, precision, recall,
                    f1_score, total_cost, confusion_matrix, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_name,
                    threshold,
                    metrics.get("roc_auc"),
                    metrics.get("pr_auc"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1"),
                    metrics.get("total_cost"),
                    json.dumps(confusion),
                    now_iso,
                ),
            )
    finally:
        conn.close()
