# llm/llm_engine.py - Layer 7: LLM Organ (Final - Tera detailed code + Production ready)

import os
from typing import Dict, Any
from openai import OpenAI  
from billing.billing import BillingLayer
from usage.usage_manager import increment_usage, get_usage


# =========================
# CONFIG FROM .env / BILLING (Production ready)
# =========================
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# Hybrid Plan ke models — exact names
OPENAI_MODELS = {
    "gpt-5.4-nano":  "gpt-5.4-nano",
    "gpt-5.4-mini":  "gpt-5.4-mini",
    "gpt-5.4":       "gpt-5.4",
    "gpt-5-nano":    "gpt-5-nano",
    "gpt-5-mini":    "gpt-5-mini",
}

# =========================
# BASE LLM INTERFACE (Agnostic) - Tera code
# =========================
class BaseLLMEngine:
    """
    Blueprint rule:
    - Brain never knows WHICH engine
    - Brain only knows WHAT it can do
    """
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLM engine must implement invoke()")

_ENGINE_CACHE = {}

def get_engine(model_name: str, max_tokens: int = 900, reasoning_effort: str = "medium"):
    """
    Hybrid Plan Quote:
    "Rule 4 — Control = Hidden
     Model switch avoid karo
     Instead: reasoning↓, tokens↓, verbosity↓"
    
    max_tokens yahan inject hota hai — tier + query type se aata hai
    """
    cache_key = f"{model_name}_{max_tokens}_{reasoning_effort}"
    if cache_key not in _ENGINE_CACHE:
        _ENGINE_CACHE[cache_key] = LLMEngineRegistry.create(
            "openai",
            model_name=model_name,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort
        )
    return _ENGINE_CACHE[cache_key]
         

# =========================
# TRANSFORMERS ENGINE - Tera detailed code
# =========================
class OpenAIEngine(BaseLLMEngine):
    """
    OpenAI API Engine — Hybrid Plan ka LLM Organ
    Brain ko model ka pata nahi — sirf invoke() call karta hai
    
    Hybrid Plan Quote:
    "Layer 2 → Query-based optimization inside phase
     Light query → downgraded model
     Medium/Heavy → primary model"
    """

    # Class level client — ek baar banao, baar baar nahi
    _client = None

    def __init__(self, model_name: str, temperature: float = 0.3,
                 max_tokens: int = 900, reasoning_effort: str = "medium"):
        self.model_name       = model_name
        self.temperature      = temperature
        self.max_tokens       = max_tokens
        self.reasoning_effort = reasoning_effort  # Issue 5 fix

        if OpenAIEngine._client is None:
            OpenAIEngine._client = OpenAI()  # env se auto-read karta hai

    def invoke(self, prompt: str) -> str:
        try:
            REASONING_MODELS = {"gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}
            extra_args = {}
            if self.model_name in REASONING_MODELS:
                extra_args["reasoning"] = {"effort": self.reasoning_effort}

            response = OpenAIEngine._client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                **extra_args
            )
            return response.output[0].content[0].text.strip()
        except Exception as e:
            # Fallback — engine crash se system crash nahi hona chahiye
            raise RuntimeError(f"[LLM Engine] API call failed: {e}")
    
# =========================
# LLM ENGINE REGISTRY - Tera code
# =========================
class LLMEngineRegistry:
    """
    Central authority for engine selection
    Brain NEVER instantiates engines directly
    """
    _registry = {}

    @classmethod
    def register(cls, name: str, engine_cls):
        cls._registry[name] = engine_cls

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"LLM Engine '{name}' not registered")
        return cls._registry[name](**kwargs)

# Register current engine

LLMEngineRegistry.register("openai", OpenAIEngine)

# Default instance


# =========================
# Public Interface for Brain/Orchestrator (Production ready)
# =========================
def generate(
    email: str,
    prompt: str,
    layer1_bundle: dict = None   # Layer 1 se aata hai — query classification ke liye
) -> str:
    """
    Brain/Orchestrator entry point
    
    Hybrid Plan complete flow yahan implement hota hai:
    
    STEP 1: Phase identify karo (usage % se)
    "Layer 1 → Time-based hard boundary
     Free/Paid: 0-70% Primary, 70-100% Secondary
     Ultra: 0-50% Primary, 50-80% Mid, 80-100% Final"
    
    STEP 2: Query type identify karo (Layer 1 bundle se)
    "Layer 2 → Query-based optimization inside phase
     complexity_score → light/medium/heavy"
    
    STEP 3: Model select karo
    "Light query → downgraded model
     Medium/Heavy → primary model"
    
    STEP 4: Token limit set karo
    "Token Control:
     Free → 700-900
     Paid → 1000-1400
     Ultra → 1300-1800"
    """

    # ===== STEP 0: Config + Usage =====
    config = BillingLayer.generate_config(email)
    tier   = config["tier"]
    limit  = config["rate_limit"]

    usage = get_usage(email)

    if limit is not None and usage >= limit:
        raise Exception("Daily query limit exceeded")

    increment_usage(email)

    # ===== STEP 1: Phase Decide (Time-based — Hybrid Plan Layer 1) =====
    usage_pct = (usage / limit * 100) if limit else 0

    phase = _get_phase(tier, usage_pct)

    # ===== STEP 2: Query Type (Layer 1 bundle se — Hybrid Plan Layer 2) =====
    query_type = _classify_query(layer1_bundle)

    # ===== STEP 3: Model + Tokens Select =====
    model_name, max_tokens, reasoning_effort = _select_model_and_tokens(
        tier, phase, query_type, layer1_bundle
    )

    # ===== STEP 4: Heavy query soft cap check =====
    # "Heavy queries: never rejected, only optimized"
    heavy_count = _get_heavy_count(email)
    heavy_limit = {"free": 5, "paid": 10, "ultra_paid": 18}.get(tier, 5)

    if query_type == "heavy" and heavy_count >= heavy_limit:
        max_tokens = int(max_tokens * 0.7)
        if reasoning_effort == "high":
            reasoning_effort = "medium"

    if query_type == "heavy":
        _increment_heavy_count(email)

    # ===== STEP 5: Engine create + Generate =====
    engine   = get_engine(model_name, max_tokens, reasoning_effort)
    response = engine.invoke(prompt)

    return response


# =========================
# HELPER FUNCTIONS — Hybrid Plan Logic
# =========================

def _get_phase(tier: str, usage_pct: float) -> str:
    """
    Hybrid Plan Quote:
    "Free/Paid: 0-70% Primary, 70-100% Secondary
     Ultra: 0-50% Primary, 50-80% Mid, 80-100% Final"
    """
    if tier in ("free", "paid"):
        return "primary" if usage_pct < 70 else "secondary"

    elif tier == "ultra_paid":
        if usage_pct < 50:   return "primary"
        elif usage_pct < 80: return "mid"
        else:                return "final"

    return "primary"  # fallback


def _classify_query(layer1_bundle: dict) -> str:
    """
    Hybrid Plan Quote:
    "Classification = compute estimation problem, not text classification"
    
    4 signals from Layer 1 (already computed — no new classifier needed):
    A) required_depth   → 40% weight  (Intent Depth)
    B) sub_goals count  → 30% weight  (Task Structure — most important)
    C) is_analytical    → 20% weight  (Reasoning Requirement)
    D) domains + graph  → 10% weight  (Context Requirement)
    """
    if not layer1_bundle:
        return "medium"  # safe fallback

    DEPTH_INDEX = {
        "shallow": 0, "normal": 1, "moderate": 2,
        "deep": 3, "very_deep": 4, "ultra_deep": 5
    }

    # Hard override — greetings
    # "word_count < 5 → force light" (sirf lower bound ke liye)
    word_count = len(layer1_bundle.get("normalized_query", "").split())
    if word_count < 3 and not layer1_bundle.get("is_analytical", False):
        return "light"

    # Signal A — Intent Depth (40%)
    depth     = layer1_bundle.get("required_depth", "normal")
    depth_idx = DEPTH_INDEX.get(depth, 1)
    sig_a     = depth_idx / 5.0

    # Signal B — Task Structure (30%) — sabse strong signal
    sub_goals = layer1_bundle.get("sub_goals", [])
    sig_b     = min(len(sub_goals) / 5.0, 1.0)

    # Signal C — Reasoning Requirement (20%)
    sig_c = 1.0 if layer1_bundle.get("is_analytical", False) else 0.0

    # Signal D — Context Requirement (10%)
    domains    = layer1_bundle.get("domains", [])
    graph_sc   = layer1_bundle.get("graph_intent_score", 0.0)
    sig_d      = min(len(domains) * 0.3 + graph_sc * 0.7, 1.0)

    # Weighted complexity score
    score = (sig_a * 0.40) + (sig_b * 0.30) + (sig_c * 0.20) + (sig_d * 0.10)

    if score < 0.35:  return "light"
    if score < 0.65:  return "medium"
    return "heavy"


def _select_model_and_tokens(tier: str, phase: str, query_type: str, layer1_bundle: dict = None):
    """
    Hybrid Plan complete routing table:
    
    FREE:
      Primary:   Light→5-nano | Med/Heavy→5.4-nano
      Secondary: All→5-nano
    
    PAID:
      Primary:   Light→5-mini | Med/Heavy→5.4-mini
      Secondary: All→5-mini
    
    ULTRA:
      Primary:   Light→5.4-mini | Med/Heavy→5.4
      Mid:       All→5.4-mini
      Final:     All→5-mini
    
    Token Control:
      Free:  Light 400-700 | Medium 700-900 | Heavy 800-900
      Paid:  Light 600-900 | Medium 1000-1300 | Heavy 1200-1400
      Ultra: Light 800-1200 | Medium 1300-1700 | Heavy 1500-1800
    """
    DEPTH_BONUS = {
        "shallow": 0, "normal": 50, "moderate": 100,
        "deep": 200, "very_deep": 300, "ultra_deep": 400
    }
    
    bundle     = layer1_bundle or {}
    depth      = bundle.get("required_depth", "normal")
    depth_add  = DEPTH_BONUS.get(depth, 0)

    # Reasoning effort — query type se
    effort_map = {"light": "low", "medium": "medium", "heavy": "high"}
    reasoning_effort = effort_map.get(query_type, "medium")

    # ── FREE ──────────────────────────────────────────────
    if tier == "free":
        if phase == "primary":
            if query_type == "light":
                return "gpt-5-nano", min(600 + depth_add, 900), "low"
            else:
                return "gpt-5.4-nano", min(800 + depth_add, 900), reasoning_effort
        else:  # secondary
            return "gpt-5-nano", 500, "low"

    # ── PAID ──────────────────────────────────────────────
    elif tier == "paid":
        if phase == "primary":
            if query_type == "light":
                return "gpt-5-mini", min(700 + depth_add, 900), "low"
            elif query_type == "medium":
                return "gpt-5.4-mini", min(1100 + depth_add, 1300), reasoning_effort
            else:  # heavy
                return "gpt-5.4-mini", min(1200 + depth_add, 1400), reasoning_effort
        else:  # secondary
            return "gpt-5-mini", 800, "low"

    # ── ULTRA ─────────────────────────────────────────────
    elif tier == "ultra_paid":
        if phase == "primary":
            if query_type == "light":
                return "gpt-5.4-mini", min(1000 + depth_add, 1200), "low"
            elif query_type == "medium":
                return "gpt-5.4", min(1400 + depth_add, 1700), reasoning_effort
            else:  # heavy
                return "gpt-5.4", min(1500 + depth_add, 1800), reasoning_effort
        elif phase == "mid":
            if query_type == "light":
                return "gpt-5-mini", 800, "low"
            return "gpt-5.4-mini", min(1200 + depth_add, 1700), reasoning_effort
        else:  # final
            return "gpt-5-mini", 900, "low"
    
    # Fallback
    return "gpt-5-nano", 500, "low"


def _get_heavy_count(email: str) -> int:
    """
    Redis se aaj ki heavy query count fetch karo
    usage_manager.py ka same pattern
    """
    from datetime import date
    import redis as _redis
    import os

    try:
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        today = date.today().isoformat()
        key   = f"heavy:{email}:{today}"
        val   = r.get(key)
        return int(val) if val else 0
    except Exception:
        return 0  # Redis down ho to safe fallback


def _increment_heavy_count(email: str):
    """
    Heavy query count badhao — 24h TTL
    """
    from datetime import date
    import redis as _redis
    import os

    try:
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        today = date.today().isoformat()
        key   = f"heavy:{email}:{today}"
        r.incr(key)
        r.expire(key, 86400)  # 24 hours — auto reset
    except Exception:
        pass  # Redis down ho to silently fail — block mat karo user ko


def get_current_config() -> Dict[str, Any]:
    return {
        "provider": "openai",
        "temperature": TEMPERATURE,
        "models": OPENAI_MODELS
    }
