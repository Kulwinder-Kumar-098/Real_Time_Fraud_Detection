"""
Flask Web Application & RESTful Risk Operations API.

Routes:
- Dashboard:               GET /
- Transaction Ledger:      GET /transactions
- Transaction Detail:      GET /transactions/<tx_id>
- Model Performance:       GET /models
- System Monitoring:       GET /monitoring
- Health Check:            GET /health

API Endpoints:
- Real-Time Scoring:       POST /api/predict
- Paginated Transactions:  GET  /api/transactions
- Single Transaction Info: GET  /api/transactions/<tx_id>
- Dashboard KPI Metrics:   GET  /api/metrics
- Live SSE Stream:         GET  /api/stream
- Simulation Controls:     POST /api/simulation/start
                           POST /api/simulation/stop
                           GET  /api/simulation/status
- Model Analytics Data:    GET  /api/model/performance
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

import config
from fraud_detection.database import (
    get_category_risk_breakdown,
    get_dashboard_kpis,
    get_recent_transactions,
    get_risk_score_distribution,
    get_transaction_by_id,
    get_volume_timeline,
    init_db,
)
from fraud_detection.inference import InferenceEngine
from fraud_detection.simulator import get_simulator

APP_START_TIME = time.time()


def create_app(test_config: Dict[str, Any] = None) -> Flask:
    """Application factory for Flask fraud detection system."""
    app = Flask(
        __name__,
        template_folder=str(config.TEMPLATES_DIR),
        static_folder=str(config.STATIC_DIR),
    )
    app.config["SECRET_KEY"] = config.SECRET_KEY
    if test_config:
        app.config.update(test_config)

    # Initialize Database safely on application setup
    with app.app_context():
        init_db()

    # Pre-warm Inference Engine & auto-start live transaction stream
    engine = InferenceEngine.get_instance()
    simulator = get_simulator()
    if not (test_config and test_config.get("TESTING")):
        simulator.start(interval=config.SIMULATION_INTERVAL_SEC)

    # ==========================================
    # HTML View Routes
    # ==========================================

    @app.route("/")
    def index():
        """Executive Risk Operations Dashboard."""
        kpis = get_dashboard_kpis()
        status = simulator.get_status()
        return render_template(
            "dashboard.html",
            kpis=kpis,
            sim_status=status,
            active_tab="dashboard",
            model_name=engine.model_name,
            threshold=engine.threshold,
        )

    @app.route("/transactions")
    def transactions_view():
        """Full transaction ledger with search, filtering, and inspection."""
        decision_filter = request.args.get("decision", "ALL")
        risk_filter = request.args.get("risk_tier", "ALL")
        search = request.args.get("q", "")
        page = max(1, int(request.args.get("page", 1)))
        per_page = 25
        offset = (page - 1) * per_page

        rows, total = get_recent_transactions(
            limit=per_page,
            offset=offset,
            decision_filter=decision_filter,
            risk_tier=risk_filter,
            search=search,
        )
        total_pages = max(1, (total + per_page - 1) // per_page)

        return render_template(
            "transactions.html",
            transactions=rows,
            current_page=page,
            total_pages=total_pages,
            total_records=total,
            decision_filter=decision_filter,
            risk_filter=risk_filter,
            search_query=search,
            active_tab="transactions",
        )

    @app.route("/transactions/<tx_id>")
    def transaction_detail_view(tx_id: str):
        """Forensic risk breakdown for an individual transaction."""
        tx = get_transaction_by_id(tx_id)
        if not tx:
            return render_template("404.html", message=f"Transaction {tx_id} not found"), 404

        return render_template(
            "transaction_detail.html",
            tx=tx,
            active_tab="transactions",
        )

    @app.route("/models")
    def models_view():
        """Model performance comparisons, ROC/PR curves, and threshold tuning."""
        metadata = {}
        if Path(config.MODEL_METADATA_PATH).exists():
            try:
                with open(config.MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        return render_template(
            "models.html",
            metadata=metadata,
            active_tab="models",
            cost_fn=config.COST_FALSE_NEGATIVE,
            cost_fp=config.COST_FALSE_POSITIVE,
        )

    @app.route("/monitoring")
    def monitoring_view():
        """Operational monitoring, latency distribution, and ground-truth validation."""
        kpis = get_dashboard_kpis()
        status = simulator.get_status()
        uptime_sec = int(time.time() - APP_START_TIME)

        return render_template(
            "monitoring.html",
            kpis=kpis,
            sim_status=status,
            uptime_sec=uptime_sec,
            model_name=engine.model_name,
            active_tab="monitoring",
        )

    # ==========================================
    # API Endpoints
    # ==========================================

    @app.route("/health", methods=["GET"])
    def health_check():
        """Deployment healthcheck endpoint."""
        db_ok = True
        try:
            get_dashboard_kpis()
        except Exception:
            db_ok = False

        status_code = 200 if (db_ok and engine.is_ready) else 200  # Return 200 with diagnostics
        return jsonify(
            {
                "status": "healthy" if db_ok else "degraded",
                "uptime_seconds": round(time.time() - APP_START_TIME, 1),
                "model_status": {
                    "is_loaded": engine.is_ready,
                    "model_name": engine.model_name,
                    "threshold": engine.threshold,
                },
                "database_connected": db_ok,
                "version": "1.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ), status_code

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        """
        Real-time transaction prediction endpoint.
        Validates input payload, scores with ML model, records to DB, and returns decision.
        """
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Invalid request. Body must be application/json."}), 400

        is_valid, err_msg, sanitized_tx = engine.validate_payload(payload)
        if not is_valid:
            return jsonify({"error": err_msg}), 400

        scored_result = engine.score_transaction(sanitized_tx, persist=True)
        return jsonify(scored_result), 200

    @app.route("/api/transactions", methods=["GET"])
    def api_transactions():
        """Returns paginated transactions with optional filters."""
        decision = request.args.get("decision", "ALL")
        risk_tier = request.args.get("risk_tier", "ALL")
        search = request.args.get("q")
        limit = min(100, max(1, int(request.args.get("limit", 25))))
        offset = max(0, int(request.args.get("offset", 0)))

        rows, total = get_recent_transactions(
            limit=limit,
            offset=offset,
            decision_filter=decision,
            risk_tier=risk_tier,
            search=search,
        )
        return jsonify({"transactions": rows, "total": total, "limit": limit, "offset": offset})

    @app.route("/api/transactions/<tx_id>", methods=["GET"])
    def api_transaction_detail(tx_id: str):
        """Returns single transaction detail."""
        tx = get_transaction_by_id(tx_id)
        if not tx:
            return jsonify({"error": f"Transaction '{tx_id}' not found."}), 404
        return jsonify(tx)

    @app.route("/api/metrics", methods=["GET"])
    def api_metrics():
        """Aggregated KPIs and chart coordinates for live dashboard polling."""
        kpis = get_dashboard_kpis()
        timeline = get_volume_timeline(limit_points=24)
        category_risk = get_category_risk_breakdown()
        risk_dist = get_risk_score_distribution()
        sim_status = simulator.get_status()

        return jsonify(
            {
                "kpis": kpis,
                "timeline": timeline,
                "category_risk": category_risk,
                "risk_distribution": risk_dist,
                "simulation": sim_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    @app.route("/api/stream")
    def api_stream():
        """Server-Sent Events (SSE) stream for real-time live transaction feed."""
        def event_stream():
            q = simulator.subscribe()
            last_heartbeat = time.time()
            try:
                # Send initial ping
                yield f"data: {json.dumps({'type': 'connected'})}\n\n"
                while True:
                    has_data = False
                    while q:
                        item = q.popleft()
                        yield f"data: {json.dumps({'type': 'transaction', 'payload': item})}\n\n"
                        has_data = True

                    now = time.time()
                    # Send keep-alive comment every 5s to keep connection open across all WSGI servers
                    if now - last_heartbeat > 5.0:
                        yield f": keep-alive {now}\n\n"
                        last_heartbeat = now

                    time.sleep(0.3)
            except GeneratorExit:
                simulator.unsubscribe(q)

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/simulation/start", methods=["POST"])
    @app.route("/api/simulator/start", methods=["POST"])
    def api_simulation_start():
        """Start or resume continuous transaction simulation."""
        data = request.get_json(silent=True) or {}
        interval = data.get("interval_sec")
        fraud_rate = data.get("fraud_rate")

        started = simulator.start(interval=interval, fraud_rate=fraud_rate)
        return jsonify(
            {
                "status": "started" if started else "already_running",
                "details": simulator.get_status(),
            }
        )

    @app.route("/api/simulation/stop", methods=["POST"])
    @app.route("/api/simulator/stop", methods=["POST"])
    def api_simulation_stop():
        """Stop or pause continuous transaction simulation."""
        stopped = simulator.stop()
        return jsonify(
            {
                "status": "stopped" if stopped else "already_stopped",
                "details": simulator.get_status(),
            }
        )

    @app.route("/api/simulation/toggle", methods=["POST"])
    @app.route("/api/simulator/toggle", methods=["POST"])
    def api_simulation_toggle():
        """Toggle continuous transaction simulation state."""
        if simulator.running:
            simulator.stop()
        else:
            data = request.get_json(silent=True) or {}
            simulator.start(interval=data.get("interval_sec"), fraud_rate=data.get("fraud_rate"))
        return jsonify(simulator.get_status())

    @app.route("/api/simulation/status", methods=["GET"])
    @app.route("/api/simulator/status", methods=["GET"])
    def api_simulation_status():
        """Current status and counters of background simulation."""
        return jsonify(simulator.get_status())

    @app.route("/api/model/performance", methods=["GET"])
    def api_model_performance():
        """Returns saved model training comparison metadata and curve data."""
        if not Path(config.MODEL_METADATA_PATH).exists():
            return jsonify({"error": "Model metadata not found. Train models first."}), 404
        with open(config.MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return jsonify(metadata)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
