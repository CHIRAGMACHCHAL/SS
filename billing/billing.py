# billing/billing.py - Layer 4: Billing & Subscription Layer (Production ready)
import os
from typing import Dict, Any

# =========================
# ENVIRONMENT VARIABLES
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required in .env file")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

class BillingLayer:
    """
    Production Billing Layer - Exact 4 Tiers as per your blueprint
    - Free, Paid, Ultra Paid (public with limits)
    - Jarvis (private, unlimited power, only for you)
    """

    @staticmethod
    def get_user_tier(email: str) -> str:
        """
        Real DB se tier fetch karega (Layer 8 ke baad)
        Abhi placeholder - aapka email jarvis tier ke liye
        """
        if email.lower() == "chirag@example.com":   # ← Aapka personal email
            return "jarvis"
        # Baaki users ke liye real DB query aayegi
        return "free"

    @staticmethod
    def generate_config(email: str) -> Dict[str, Any]:
        """
        Blueprint ke hisaab se exact config banata hai
        Jarvis = unlimited power
        Public tiers = achha power lekin controlled
        """
        tier = BillingLayer.get_user_tier(email)

        if tier == "jarvis":
            return {
                "tier": "jarvis",
                "max_docs": 999999,           # Practically unlimited
                "deep_reasoning": True,
                "use_emergent_concepts": True,
                "query_complexity": "high",
                "allow_agency": True,         # Full agency
                "allow_self_training": True,  # Phase 6 enabled
                "max_tokens": 32768,          # Very high (model limit ke andar)
                "collection": "jarvis_private", # Private memory
                "allowed_tools": ["web_search", "code_execution", "file_access", "external_api", "all"],
                "private_memory": True,
                "long_term_memory": True,
                "unlimited_mode": True
            }

        elif tier == "ultra_paid":
            return {
                "tier": "ultra_paid",
                "max_docs": 25,
                "deep_reasoning": True,
                "use_emergent_concepts": True,
                "query_complexity": "high",
                "allow_agency": False,        # Limited agency (misuse prevention)
                "allow_self_training": False,
                "max_tokens": 12000,
                "collection": "ultra_memory",
                "allowed_tools": ["web_search", "code_execution"],
                "private_memory": False
            }

        elif tier == "paid":
            return {
                "tier": "paid",
                "max_docs": 12,
                "deep_reasoning": True,
                "use_emergent_concepts": True,
                "query_complexity": "normal",
                "allow_agency": False,
                "allow_self_training": False,
                "max_tokens": 6000,
                "collection": "pro_memory",
                "allowed_tools": ["web_search"],
                "private_memory": False
            }

        else:  # free
            return {
                "tier": "free",
                "max_docs": 6,
                "deep_reasoning": False,
                "use_emergent_concepts": False,
                "query_complexity": "low",
                "allow_agency": False,
                "allow_self_training": False,
                "max_tokens": 3000,
                "collection": "public_core",
                "allowed_tools": [],
                "private_memory": False
            }