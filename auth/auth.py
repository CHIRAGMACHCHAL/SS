import hashlib
import hmac
import time
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

# Secrets ( .env se lenge )
SECRET_KEY = os.getenv("SECRET_KEY", "your-very-long-secret-key-change-in-production")
TOKEN_EXPIRE_MINUTES = 1440  # 24 hours for production

class AuthEngine:
    """
    Professional Authentication Layer - No external auth library used for core logic
    Supports: Email/Password + JWT + Future OAuth
    Secure: PBKDF2 + HMAC + Constant time compare
    """

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """PBKDF2 hashing - industry standard (like Django/Flask)"""
        if salt is None:
            salt = os.urandom(32)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return hashed, salt

    @staticmethod
    def verify_password(stored_hash: bytes, salt: bytes, provided_password: str) -> bool:
        """Constant time comparison - security best practice"""
        computed_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode(), salt, 100000)
        return hmac.compare_digest(computed_hash, stored_hash)

    @staticmethod
    def create_jwt(payload: Dict) -> str:
        """Custom JWT (no pyjwt library - pure Python)"""
        header = {"alg": "HS256", "typ": "JWT"}
        expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
        payload["exp"] = int(expire.timestamp())
        
        # Base64 encode (manual)
        import base64
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip('=')
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip('=')
        
        # Signature
        signing_input = f"{header_b64}.{payload_b64}".encode()
        signature = hmac.new(SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        return f"{header_b64}.{payload_b64}.{signature_b64}"

    @staticmethod
    def verify_jwt(token: str) -> Optional[Dict]:
        """Verify JWT - production ready"""
        try:
            header_b64, payload_b64, signature_b64 = token.split('.')
            
            # Re-create signature
            signing_input = f"{header_b64}.{payload_b64}".encode()
            signature = hmac.new(SECRET_KEY.encode(), signing_input, hashlib.sha256).digest()
            expected_sig = base64.urlsafe_b64encode(signature).decode().rstrip('=')
            
            if not hmac.compare_digest(expected_sig, signature_b64):
                return None
                
            # Decode payload
            import base64
            payload_json = base64.urlsafe_b64decode(payload_b64 + '==').decode()
            payload = json.loads(payload_json)
            
            # Check expiry
            if payload.get("exp", 0) < time.time():
                return None
                
            return payload
        except:
            return None

    @staticmethod
    def generate_token(user_id: str, email: str, tier: str) -> str:
        payload = {
            "sub": user_id,
            "email": email,
            "tier": tier,
            "iat": int(time.time())
        }
        return AuthEngine.create_jwt(payload)

# Example usage (test ke liye)
if __name__ == "__main__":
    # Test password hash
    hashed, salt = AuthEngine.hash_password("MyStrongPassword123")
    print("Password hashed successfully")
    
    # Test token
    token = AuthEngine.generate_token("user_123", "chirag@example.com", "ultra")
    print("Token generated:", token[:50] + "...")
    
    # Verify
    decoded = AuthEngine.verify_jwt(token)
    print("Token verified:", decoded is not None)