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

                "response_strategy_mode": "jarvis",
                "capability_level": "jarvis",
                "cognitive_load_level": "maximum",
                "meta_retry_enabled": True,
                "meta_confidence_threshold": "medium",  # medium se neeche retry
                "meta_self_critical": True,   # Jarvis only — extra flag

                "memory_scope":           "all",
                "identity_reinforcement": True,

                "allow_research":          True,
                "allow_analysis":          True,
                "allow_execution":         True,   # full power
                "allow_agency":            True,   # full agency
                "allow_ancient_tech":      True,   # Viman, Vedic, Astra decode
                "sensitive_domain_caution":True,  # calculated risk — no blind blocks
                "allow_dangerous_keywords": True,
                "allow_sensitive_override": True,   # Jarvis ko block nahi karega sensitive domain bhi

                "allow_goal_inference":       True,
                "allow_plan_synthesis":       True,
                "allow_action_selection":     True,
                "allow_tool_invocation":      True,
                "allow_execution_monitoring": True,

                "max_docs": 999999,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "maximum",  # Jarvis = 8 goals — full power    
                "allow_self_training": True,  
                "max_tokens": 32768,  
                "collection": "jarvis_private",  
                "allowed_tools": ["web_search", "code_execution", "file_access", "external_api", "ancient_tech_decoder", "virtual_simulation", "temple_geometry_analyzer", "scripture_cross_reference", "ancient_modern_blend", "pdf_deep_analysis"],  
                "private_memory": True,  
                "long_term_memory": True,  
                "unlimited_mode": True,  
                "conversation_history_limit": 100,
                "trace_logging": True,  
                "rate_limit": None  # Unlimited  
            }  
  
        # ========== TIER 5: ENTERPRISE (LARGE COMPANIES) ==========  
        elif tier == "enterprise":  
            return {  
                "tier": "enterprise", 

                "response_strategy_mode": "expert",
                "capability_level": "expert",
                "cognitive_load_level": "expert",
                "meta_retry_enabled": True,
                "meta_confidence_threshold": "medium",

                "allow_research":          True,
                "allow_analysis":          True,
                "allow_execution":         False,  # large org mein AGI execution nahi
                "allow_agency":            False,
                "allow_ancient_tech":      False,
                "sensitive_domain_caution":True,   # extra — compliance critical
                "allow_sensitive_override": False,  # Enterprise block hoga sensitive domain pe

                "allow_goal_inference":       False,
                "allow_plan_synthesis":       False,
                "allow_action_selection":     False,
                "allow_tool_invocation":      False,
                "allow_execution_monitoring": False,

                "memory_scope":           "public_only",
                "identity_reinforcement": False,

                "max_docs": 50,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "very_high",    
                "allow_self_training": False,  
                "max_tokens": 20000,  
                "collection": "public_core",  
                "allowed_tools": ["web_search", "code_execution", "pdf_deep_analysis"],  
                "private_memory": False,  
                "long_term_memory": True,  
                "unlimited_mode": False,  
                "conversation_history_limit": 50,
                "trace_logging": False,  
                "rate_limit": 1000  # requests per day  
            }  
  
        # ========== TIER 4: BUSINESS SMALL (SMALL COMPANIES) ==========  
        elif tier == "business_small":  
            return {  
                "tier": "business_small",  

                "response_strategy_mode": "professional",
                "capability_level": "professional",
                "cognitive_load_level": "professional",
                "meta_retry_enabled": True,
                "meta_confidence_threshold": "low",  # sirf low pe retry

                "allow_research":          True,   # company ko research chahiye
                "allow_analysis":          True,
                "allow_execution":         False,  # execution sensitive hai company mein
                "allow_agency":            False,
                "allow_ancient_tech":      False,
                "sensitive_domain_caution":False,

                "allow_goal_inference":       False,
                "allow_plan_synthesis":       False,
                "allow_action_selection":     False,
                "allow_tool_invocation":      False,
                "allow_execution_monitoring": False,

                "memory_scope":           "public_only",
                "identity_reinforcement": False,

                "max_docs": 35,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "high",   
                "allow_self_training": False,  
                "max_tokens": 15000,  
                "collection": "public_core",  
                "allowed_tools": ["web_search", "code_execution"],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 30,
                "trace_logging": False,  
                "rate_limit": 500  # requests per day  
            }  
  
        # ========== TIER 3: ULTRA PAID (INDIVIDUAL) ==========  
        elif tier == "ultra_paid":  
            return {  
                "tier": "ultra_paid",
  
                "response_strategy_mode": "advanced",
                "capability_level": "advanced",
                "cognitive_load_level": "advanced",
                "meta_retry_enabled": True,
                "meta_confidence_threshold": "low",
                
                "allow_research":          False,  # individual hai — research nahi
                "allow_analysis":          True,   # complex analysis milti hai
                "allow_execution":         False,
                "allow_agency":            False,
                "allow_ancient_tech":      False,
                "sensitive_domain_caution":False,

                "allow_goal_inference":       False,
                "allow_plan_synthesis":       False,
                "allow_action_selection":     False,
                "allow_tool_invocation":      False,
                "allow_execution_monitoring": False,

                "memory_scope":           "public_only",
                "identity_reinforcement": False,


                "max_docs": 25,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "high",  
                "allow_self_training": False,  
                "max_tokens": 12000,  
                "collection": "public_core",  
                "allowed_tools": ["web_search", "code_execution"],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 20, 
                "trace_logging": False, 
                "rate_limit": 200  # requests per day  
            }  
  
        # ========== TIER 2: PAID (INDIVIDUAL) ==========  
        elif tier == "paid":  
            return {  
                "tier": "paid", 
 
                "response_strategy_mode": "standard",
                "capability_level": "standard",
                "cognitive_load_level": "standard",
                "meta_retry_enabled": True,
                "meta_confidence_threshold": "low",

                "allow_research":          False,  # individual hai, research project nahi
                "allow_analysis":          False,  # abhi basic level
                "allow_execution":         False,
                "allow_agency":            False,
                "allow_ancient_tech":      False,
                "sensitive_domain_caution":False,

                "allow_goal_inference":       False,
                "allow_plan_synthesis":       False,
                "allow_action_selection":     False,
                "allow_tool_invocation":      False,
                "allow_execution_monitoring": False,

                "memory_scope":           "public_only",
                "identity_reinforcement": False,
 


                "max_docs": 12,  
                "deep_reasoning": True,  
                "use_emergent_concepts": True,  
                "query_complexity": "normal",  
                "allow_self_training": False,  
                "max_tokens": 6000,  
                "collection": "public_core",  
                "allowed_tools": ["web_search"],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 10,  
                "trace_logging": False,
                "rate_limit": 100  # requests per day  
            }  
  
        # ========== TIER 1: FREE (INDIVIDUAL) ==========  
        else:  # free  
            return {  
                "tier": "free",

                "response_strategy_mode": "basic",#2-----------------------------
                "capability_level": "base", #3-------------------
                "cognitive_load_level": "minimal",#1----------------------------
                "meta_retry_enabled": True,#4--------------------  # Free users ke liye retry enabled hai, par sirf low confidence pe 
                "meta_confidence_threshold": "low", #5--------------------------

                "allow_research":          False,#12--------------------  # researcher nahi hai
                "allow_analysis":          False,#13----------------  # complex analysis nahi chahiye
                "allow_execution":         False,#14-------------------------------
                "allow_agency":            False, #6-----------------------------------------
                "allow_ancient_tech":      False,#15---------------------------------------
                "sensitive_domain_caution":False,

                "allow_goal_inference":       False, #7---------------------------------------
                "allow_plan_synthesis":       False,#8---------------------------------------
                "allow_action_selection":     False,#9--------------------------------------
                "allow_tool_invocation":      False,#10-------------------------------------
                "allow_execution_monitoring": False,#11----------------------------------

                "memory_scope":           "public_only",
                "identity_reinforcement": False,



                "max_docs": 6,  
                "deep_reasoning": False,  
                "use_emergent_concepts": False,  
                "query_complexity": "low",    
                "allow_self_training": False,  
                "max_tokens": 3000,  
                "collection": "public_core",  
                "allowed_tools": [],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 10, 
                "trace_logging": False, 
                "rate_limit": 50  # requests per day  
            }