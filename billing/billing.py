# billing/billing.py - Layer 4: Billing & Subscription Layer (Production ready)  
import os  
from typing import Dict, Any, List  
  
# =========================  
# ENVIRONMENT VARIABLES  
# =========================  
DATABASE_URL = os.getenv("DATABASE_URL", "test://localhost")  # Default for testing
if not DATABASE_URL:  
    raise RuntimeError("DATABASE_URL is required in .env file")  
  
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")  
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")  
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")  



def get_model_for_user(email: str) -> str:
    tier = BillingLayer.get_user_tier(email)

    from usage.usage_manager import get_usage

    usage = get_usage(email)
    config = BillingLayer.generate_config(email)
    limit = config["rate_limit"]

    # Unlimited users
    if limit is None:
        return "gpt-5.4"

    # ===== PRECOMPUTED THRESHOLDS =====
    t70 = int(limit * 0.7)
    t50 = int(limit * 0.5)
    t80 = int(limit * 0.8)

    # ===== FREE =====
    if tier == "free":
        if usage < t70:
            return "gpt-5.4-nano"
        else:
            return "gpt-5-nano"

    # ===== PAID =====
    elif tier == "paid":
        if usage < t70:
            return "gpt-5.4-mini"
        else:
            return "gpt-5-mini"

    # ===== ULTRA =====
    elif tier == "ultra_paid":
        if usage < t50:
            return "gpt-5.4"
        elif usage < t80:
            return "gpt-5.4-mini"
        else:
            return "gpt-5-mini"

    return "gpt-5.4-nano"  
class BillingLayer:  
    """  
    Production Billing Layer - 6 Tiers System  
    - Individual Tiers: Free, Paid, Ultra Paid (public)  
    - Business Tiers: Business Small, Enterprise (companies)  
    - Jarvis: Private, unlimited power (founder only)  
    """  
    
    # Modality Limits per Tier (Industry Competitive)
    TIER_MODALITY_LIMITS = {
        'free': {
            'modalities': ['text', 'image', 'document', 'code'],
            'max_files': 3,
            'max_file_size_mb': 10,
            'max_total_size_mb': 25,
            'max_text_tokens': 50000,
            'context_window_tokens': 32000,
            'image_limit_mb': 5,
            'image_count_per_day': 5,
            'max_goals': 2,
            'ocr_enabled': False,        # ❌ Free tier me OCR nahi
            'asr_enabled': False         # ❌ Free tier me ASR nahi
        },
        'paid': {
            'modalities': ['text', 'image', 'document', 'code', 'audio', 'data'],
            'max_files': 10,
            'max_file_size_mb': 50,
            'max_total_size_mb': 200,
            'max_text_tokens': 200000,
            'context_window_tokens': 64000,
            'image_limit_mb': 20,
            'image_count_per_day': 20,
            'audio_limit_mb': 50,
            'max_goals': 5,
            'ocr_enabled': True,         # ✅ Paid tier me basic OCR
            'asr_enabled': False         # ❌ Paid tier me ASR nahi
        },
        'ultra_paid': {
            'modalities': ['text', 'image', 'document', 'code', 'audio', 'data', 'voice'],
            'max_files': 25,
            'max_file_size_mb': 100,
            'max_total_size_mb': 500,
            'max_text_tokens': 500000,
            'context_window_tokens': 128000,
            'image_limit_mb': 50,
            'image_count_per_day': 50,
            'audio_limit_mb': 100,
            'voice_limit_mb': 50,
            'max_goals': 8,
            'ocr_enabled': True,         # ✅ Full OCR
            'asr_enabled': True,         # ✅ Basic ASR enabled
        },
        'business_small': {
            'modalities': ['text', 'image', 'document', 'code', 'audio', 'data', 'voice', 'video'],
            'max_files': 50,
            'max_file_size_mb': 200,
            'max_total_size_mb': 1024,
            'max_text_tokens': 1000000,
            'context_window_tokens': 256000,
            'image_limit_mb': 100,
            'image_count_per_day': 100,
            'audio_limit_mb': 200,
            'voice_limit_mb': 100,
            'video_limit_mb': 500,
            'max_goals': 12,
            'ocr_enabled': True,         # ✅ Full OCR
            'asr_enabled': True,         # ✅ Full ASR
        },
        'enterprise': {
            'modalities': ['text', 'image', 'document', 'code', 'audio', 'data', 'voice', 'video', 'binary'],
            'max_files': 100,
            'max_file_size_mb': 500,
            'max_total_size_mb': 2048,
            'max_text_tokens': 2000000,
            'context_window_tokens': 512000,
            'image_limit_mb': 200,
            'image_count_per_day': -1,  # Unlimited
            'audio_limit_mb': 500,
            'voice_limit_mb': 200,
            'video_limit_mb': 1024,
            'binary_limit_mb': 1024,
            'max_goals': 20,
            'ocr_enabled': True,         # ✅ Full OCR
            'asr_enabled': True,         # ✅ Full ASR
        },
        'jarvis': {
            'modalities': ['text', 'image', 'document', 'code', 'audio', 'data', 'voice', 'video', 'binary', 'multimodal'],
            'max_files': -1,  # Unlimited
            'max_file_size_mb': 1024,
            'max_total_size_mb': 5120,
            'max_text_tokens': 5000000,
            'context_window_tokens': 1000000,
            'image_limit_mb': 500,
            'image_count_per_day': -1,  # Unlimited
            'audio_limit_mb': 1024,
            'voice_limit_mb': 500,
            'video_limit_mb': 2048,
            'binary_limit_mb': 2048,
            'max_goals': -1,  # Unlimited
            'ocr_enabled': True,         # ✅ Full Power
            'asr_enabled': True,         # ✅ Full Power
        }
    }  
  
    @staticmethod
    def get_modality_limits(email: str) -> Dict[str, Any]:
        """
        Get modality limits for user tier
        Integration with Phase 1.0 signal capture
        """
        tier = BillingLayer.get_user_tier(email)
        return BillingLayer.TIER_MODALITY_LIMITS.get(tier, BillingLayer.TIER_MODALITY_LIMITS['free'])
    
    @staticmethod
    def validate_modality_access(email: str, requested_modalities: List[str]) -> Dict[str, Any]:
        """
        Validate modality access against tier limits
        Returns validation result for Phase 1.0
        """
        tier_limits = BillingLayer.get_modality_limits(email)
        allowed_modalities = tier_limits['modalities']
        
        validation_result = {
            'tier': BillingLayer.get_user_tier(email),
            'allowed_modalities': allowed_modalities,
            'requested_modalities': requested_modalities,
            'access_granted': True,
            'blocked_modalities': []
        }
        
        # Check each requested modality
        for modality in requested_modalities:
            if modality not in allowed_modalities:
                validation_result['access_granted'] = False
                validation_result['blocked_modalities'].append(modality)
        
        return validation_result

    @staticmethod  
    def get_user_tier(email: str) -> str:  
        """  
        Real DB se tier fetch karega (Layer 8 ke baad)  
        Abhi placeholder - aapka email jarvis tier ke liye  
        """  
        if email.lower() == "chirag@example.com":  # ← Aapka personal email
            return "jarvis"  
        elif email.lower() == "enterprise@example.com":
            return "enterprise"
        elif email.lower() == "business@example.com":
            return "business_small"
        elif email.lower() == "ultra@example.com":
            return "ultra_paid"
        elif email.lower() == "paid@example.com":
            return "paid"
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

                "max_goals": 100,    # jarvis 
                "max_query_expansion": 100,  # jarvis
                "max_depth": "ultra_deep",
                "max_nlp_power": 100,


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

                "max_goals": 15,     # enterprise
                "max_query_expansion": 8,    # enterprise
                "max_depth": "very_deep",
                "max_nlp_power": 60,

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
                "query_complexity": "expert",    
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

                "max_goals": 10,     # business 
                "max_query_expansion": 5,    # business
                "max_depth": "deep",
                "max_nlp_power": 35,

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

                "max_goals": 6,      # ultra_paid
                "max_query_expansion": 3,    # ultra_paid
                "max_depth": "moderate",
                "max_nlp_power": 20,
  
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
                # "max_tokens": 12000,
                "max_tokens": 3000,  
                "collection": "public_core",  
                "allowed_tools": ["web_search", "code_execution"],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 20, 
                "trace_logging": False, 
                "rate_limit": 60  # requests per day  
            }  
  
        # ========== TIER 2: PAID (INDIVIDUAL) ==========  
        elif tier == "paid":  
            return {  
                "tier": "paid",

                "max_goals": 4,      # paid 
                "max_query_expansion": 2,    # paid
                "max_depth": "normal",
                "max_nlp_power": 10,
 
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
                # "max_tokens": 6000,
                "max_tokens": 1500,  
                "collection": "public_core",  
                "allowed_tools": ["web_search"],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 10,  
                "trace_logging": False,
                "rate_limit": 40  # requests per day  
            }  
  
        # ========== TIER 1: FREE (INDIVIDUAL) ==========  
        else:  # free  
            return {  
                "tier": "free",

                "max_goals": 2,      # free
                "max_query_expansion": 1,    # free
                "max_depth": "shallow",
                "max_nlp_power": 5,

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
                # "max_tokens": 3000,
                "max_tokens": 500,  
                "collection": "public_core",  
                "allowed_tools": [],  
                "private_memory": False,  
                "long_term_memory": False,  
                "unlimited_mode": False,  
                "conversation_history_limit": 10, 
                "trace_logging": False, 
                "rate_limit": 25  # requests per day  
            }

            
# 🧠 STEP 7 — SAFETY (warna loss hoga)
# Add this:
# ❌ Hard stop:
# if monthly_usage > limit:
#     block or downgrade model
# ✅ Soft throttle:
# after 70% usage:
#     switch to cheaper model            





# 🧮 Assume:
# Paid user:
# avg 40 queries/day use karega (not 80)
# 40 × 30 = 1200 queries
# cost ≈ ₹600
# revenue = ₹199

# 👉 loss? ❌

# Real behavior:

# 👉 80% users:

# 10–30 queries/day use karte hain
# avg = 20 queries/day
# monthly = 600 queries

# cost ≈ ₹300
# revenue = ₹199

# 👉 slight loss per heavy user
# 👉 but overall profit due to:

# 🔥 Hidden profit drivers
# 1. Low usage users
# 50% users use <10 queries/day
# 👉 high profit
# 2. Upgrade funnel
# paid → ultra
# 3. Free users convert
# 🚀 STEP 5 — ULTIMATE PROFIT TRICK (important)
# Dynamic cost control

# 👉 Same tier me bhi:

# if simple query:
#     use nano

# if medium:
#     use mini

# if complex:
#     use full

# 👉 Isse:

# cost 30–50% reduce
# profit double
# ⚠️ STEP 6 — LIMITS TUNING (final)
# Replace this in your code:
# FREE:
# rate_limit = 20

# PAID:
# rate_limit = 80

# ULTRA:
# rate_limit = 150
# Token control add karo (VERY IMPORTANT)
# max_tokens_per_response:

# free = 500
# paid = 1500
# ultra = 3000


# 🔥 Ek pro-level advice

# 👉 Agar tum ye add kar do:

# “Top-up credits” system

# Example:

# ₹50 = extra 100 queries

# 👉 Profit aur stable ho jayega