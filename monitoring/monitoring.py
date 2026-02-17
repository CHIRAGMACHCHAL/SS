import os
import time
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import prometheus_client as prom
from prometheus_client import Counter, Gauge, Histogram, Summary

# =========================
# PROMETHEUS METRICS (Industry Standard)
# =========================
REQUESTS_TOTAL = Counter(
    'vedic_agi_requests_total', 'Total requests', ['tier', 'endpoint']
)
REQUEST_LATENCY = Histogram(
    'vedic_agi_request_latency_seconds', 'Request latency', ['tier', 'endpoint']
)
ERRORS_TOTAL = Counter(
    'vedic_agi_errors_total', 'Total errors', ['tier', 'error_type']
)
TOKENS_USED = Gauge(
    'vedic_agi_tokens_used', 'Tokens used per request', ['tier']
)
ACTIVE_USERS = Gauge(
    'vedic_agi_active_users', 'Active users count', ['tier']
)

# =========================
# LOGGING SETUP (Production Ready)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Future mein file handler add kar sakte hain
    ]
)
logger = logging.getLogger("VedicAGI_Monitoring")

class MonitoringLayer:
    """
    Production Monitoring & Logging Layer (Layer 10)
    - Blueprint + Goal ke hisaab se powerful tracking
    - Prometheus metrics for dashboard (latency, usage, errors)
    - Detailed logs for Jarvis tier
    - Usage tracking for billing & abuse detection
    - No dummy — real metrics & logs
    """

    def __init__(self):
        self.tier_metrics = {}
        self.start_time = None

    def start_request(self, tier: str, endpoint: str = "/chat"):
        """Request start hone pe call karo"""
        self.start_time = time.time()
        REQUESTS_TOTAL.labels(tier=tier, endpoint=endpoint).inc()
        ACTIVE_USERS.labels(tier=tier).inc()
        logger.info(f"[START] Request | Tier: {tier} | Endpoint: {endpoint}")

    def end_request(self, tier: str, endpoint: str, tokens_used: int = 0, error: Optional[str] = None):
        """Response dene se pehle call karo"""
        if self.start_time is None:
            return

        latency = time.time() - self.start_time
        REQUEST_LATENCY.labels(tier=tier, endpoint=endpoint).observe(latency)

        if tokens_used > 0:
            TOKENS_USED.labels(tier=tier).set(tokens_used)

        if error:
            ERRORS_TOTAL.labels(tier=tier, error_type=error).inc()
            logger.error(f"[ERROR] Request failed | Tier: {tier} | Error: {error} | Latency: {latency:.3f}s")
        else:
            logger.info(f"[SUCCESS] Request completed | Tier: {tier} | Latency: {latency:.3f}s | Tokens: {tokens_used}")

        ACTIVE_USERS.labels(tier=tier).dec()
        self.start_time = None

    def log_jarvis_trace(self, trace_data: Dict[str, Any]):
        """Jarvis tier ke liye detailed trace log"""
        if trace_data.get("tier") != "jarvis":
            return

        trace_json = json.dumps(trace_data, indent=2, ensure_ascii=False)
        logger.info(f"[JARVIS TRACE]\n{trace_json}")

    def expose_metrics(self):
        """Prometheus endpoint ke liye (future deployment mein use)"""
        return prom.generate_latest()

# Global instance (Orchestrator/Brain se import karenge)
monitoring = MonitoringLayer()