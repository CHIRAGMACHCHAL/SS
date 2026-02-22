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
    Production Billing Layer - 6 Tiers System  
    - Individual Tiers: Free, Paid, Ultra Paid (public)  
    - Business Tiers: Business Small, Enterprise (companies)  
    - Jarvis: Private, unlimited power (founder only)  
    """  
  
    @staticmethod  
    def get_user_tier(email: str) -> str:  
        """  
        Real DB se tier fetch karega (Layer 8 ke baad)  
        Abhi placeholder - aapka email jarvis tier ke liye  
        """  
        if email.lower() == "chirag@example.com":  # ← Aapka personal email
            return "jarvis"  
        # TODO: Real DB query for production  
        # Example: SELECT tier FROM users WHERE email = $1  
        return "free"  
  
    @staticmethod  
    def generate_config(email: str) -> Dict[str, Any]:  
        """  
        6-Tier configuration generator  
        Industry-standard pattern with capability differentiation  
        """  
        tier = BillingLayer.get_user_tier(email)  
  
        # ========== TIER 6: JARVIS (FOUNDER) ==========  
        if tier == "jarvis":  
            return {  
                "tier": "jarvis",  
                "max_docs": 999999,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "high",  
                "allow_agency": True,  
                "allow_self_training": True,  
                "max_tokens": 32768,  
                "collection": "jarvis_private",  
                "allowed_tools": ["web_search", "code_execution", "file_access", "external_api", "ancient_tech_decoder", "virtual_simulation", "temple_geometry_analyzer", "scripture_cross_reference", "ancient_modern_blend", "pdf_deep_analysis"],  
                "private_memory": True,  
                "long_term_memory": True,  
                "unlimited_mode": True,  
                "conversation_history_limit": 100,  
                "rate_limit": None  # Unlimited  
            }  
  
        # ========== TIER 5: ENTERPRISE (LARGE COMPANIES) ==========  
        elif tier == "enterprise":  
            return {  
                "tier": "enterprise",  
                "max_docs": 50,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "high",  
                "allow_agency": False,  
                "allow_self_training": False,  
                "max_tokens": 20000,  
                "collection": "public_core",  
                "allowed_tools": ["web_search", "code_execution", "pdf_deep_analysis"],  
                "private_memory": False,  
                "long_term_memory": True,  
                "unlimited_mode": False,  
                "conversation_history_limit": 50,  
                "rate_limit": 1000  # requests per day  
            }  
  
        # ========== TIER 4: BUSINESS SMALL (SMALL COMPANIES) ==========  
        elif tier == "business_small":  
            return {  
                "tier": "business_small",  
                "max_docs": 35,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "high",  
                "allow_agency": False,  
                "allow_self_training": False,  
                "max_tokens": 15000,  
                "collection": "public_core",  
                "allowed_tools": ["web_search", "code_execution"],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 30,  
                "rate_limit": 500  # requests per day  
            }  
  
        # ========== TIER 3: ULTRA PAID (INDIVIDUAL) ==========  
        elif tier == "ultra_paid":  
            return {  
                "tier": "ultra_paid",  
                "max_docs": 25,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "high",  
                "allow_agency": False,  
                "allow_self_training": False,  
                "max_tokens": 12000,  
                "collection": "public_core",  
                "allowed_tools": ["web_search", "code_execution"],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 20,  
                "rate_limit": 200  # requests per day  
            }  
  
        # ========== TIER 2: PAID (INDIVIDUAL) ==========  
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
                "collection": "public_core",  
                "allowed_tools": ["web_search"],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 10,  
                "rate_limit": 100  # requests per day  
            }  
  
        # ========== TIER 1: FREE (INDIVIDUAL) ==========  
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
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 10,  
                "rate_limit": 50  # requests per day  
            }