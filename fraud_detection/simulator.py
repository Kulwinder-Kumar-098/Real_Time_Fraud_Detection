"""
Live Transaction Stream Simulator for Real-Time Fraud Operations.

Generates continuous synthetic transactions in a background daemon thread,
routes them through the InferenceEngine for scoring, and broadcasts them
via an in-memory queue to Server-Sent Events (SSE) subscribers.
"""

from collections import deque
import json
import random
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional

import config
from fraud_detection.database import get_all_users
from fraud_detection.generator import (
    SyntheticTransactionGenerator,
    generate_user_profiles,
)
from fraud_detection.inference import InferenceEngine


class StreamingSimulator:
    _instance: Optional["StreamingSimulator"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.interval = config.SIMULATION_INTERVAL_SEC
        self.fraud_rate = config.SIMULATION_FRAUD_RATE
        self.recent_stream: deque = deque(maxlen=60)
        self.subscribers: List[deque] = []
        self._sub_lock = threading.Lock()

        # Generator & Inference Engine
        self.engine = InferenceEngine.get_instance()
        profiles = get_all_users()
        if not profiles:
            profiles = generate_user_profiles(150, seed=config.RANDOM_SEED)
        self.generator = SyntheticTransactionGenerator(user_profiles=profiles)

        self.stats = {
            "total_generated": 0,
            "fraud_generated": 0,
            "declined_count": 0,
            "reviewed_count": 0,
            "approved_count": 0,
            "start_time": None,
        }

    @classmethod
    def get_instance(cls) -> "StreamingSimulator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = StreamingSimulator()
            return cls._instance

    def start(self, interval: Optional[float] = None, fraud_rate: Optional[float] = None) -> bool:
        """Starts background transaction generation thread."""
        with self._lock:
            if self.running:
                return False

            if interval is not None:
                self.interval = max(0.2, float(interval))
            if fraud_rate is not None:
                self.fraud_rate = max(0.0, min(1.0, float(fraud_rate)))

            self._stop_event.clear()
            self.running = True
            if self.stats["start_time"] is None:
                self.stats["start_time"] = time.time()

            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            print(f"[Simulator] Started live transaction stream (interval={self.interval}s, fraud_rate={self.fraud_rate:.1%})")
            return True

    def stop(self) -> bool:
        """Stops background transaction generation thread."""
        with self._lock:
            if not self.running:
                return False
            self.running = False
            self._stop_event.set()
            print("[Simulator] Stopped live transaction stream.")
            return True

    def get_status(self) -> Dict[str, Any]:
        """Returns operational state and simulation counters."""
        return {
            "is_running": self.running,
            "interval_sec": self.interval,
            "fraud_rate": self.fraud_rate,
            "stats": self.stats,
            "recent_count": len(self.recent_stream),
        }

    def subscribe(self) -> deque:
        """Subscribes an SSE client listener queue."""
        q: deque = deque(maxlen=20)
        with self._sub_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: deque) -> None:
        """Removes an SSE client listener queue."""
        with self._sub_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _broadcast(self, item: Dict[str, Any]) -> None:
        """Pushes scored transaction to all active client queues."""
        self.recent_stream.appendleft(item)
        with self._sub_lock:
            for q in self.subscribers:
                q.append(item)

    def _run_loop(self) -> None:
        """Continuous background transaction synthesis and real-time scoring."""
        while not self._stop_event.is_set():
            try:
                # 1. Synthesize transaction
                is_fraud = (random.random() < self.fraud_rate)
                raw_tx = self.generator.generate_single_transaction(force_fraud=is_fraud)

                # 2. Score via Inference Engine
                scored_tx = self.engine.score_transaction(raw_tx, persist=True)

                # 3. Update simulation counters
                self.stats["total_generated"] += 1
                if scored_tx.get("ground_truth_label") == 1:
                    self.stats["fraud_generated"] += 1

                dec = scored_tx.get("decision", "APPROVE")
                if dec == "DECLINE":
                    self.stats["declined_count"] += 1
                elif dec == "REVIEW":
                    self.stats["reviewed_count"] += 1
                else:
                    self.stats["approved_count"] += 1

                # 4. Broadcast to live connected UI clients
                self._broadcast(scored_tx)

            except Exception as e:
                print(f"[Simulator] Error in simulation loop: {e}")

            # Sleep interval with interruptible check
            self._stop_event.wait(timeout=self.interval)


def get_simulator() -> StreamingSimulator:
    return StreamingSimulator.get_instance()
