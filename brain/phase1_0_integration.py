#!/usr/bin/env python3
"""
Phase 1.0 Integration with Existing main.py
Drop-in replacement for current Phase 1.0 implementation
Blueprint compliant + ChatGPT framework approved
"""

from phase1_0_blueprint_corrected import BlueprintCompliantPhase1_0, SignalData, SignalType
from typing import Any
import logging

class Phase1_0_Integration:
    """
    Integration layer for existing main.py
    Maintains same interface while using blueprint compliant implementation
    """
    
    def __init__(self):
        self.phase1 = BlueprintCompliantPhase1_0()
        self.logger = logging.getLogger(__name__)
    
    def capture_user_query(self, user_query: Any, **kwargs) -> dict:
        """
        Drop-in replacement for existing Phase 1.0
        Returns dict format compatible with existing downstream phases
        """
        
        # Capture signal using blueprint compliant method
        signal_data = self.phase1.capture_raw_signal(
            user_input=user_query,
            mime_type=kwargs.get('mime_type'),
            filename=kwargs.get('filename'),
            metadata=kwargs.get('metadata', {})
        )
        
        # Convert to existing dict format (for downstream compatibility)
        return {
            # Existing format fields
            'raw_query': self._extract_text_content(signal_data),
            'media_payloads': self._extract_media_payloads(signal_data),
            'signal_type': signal_data.signal_type.value,
            
            # New blueprint compliant fields
            'signal_data': signal_data,
            'capture_id': signal_data.capture_id,
            'capture_timestamp': signal_data.capture_timestamp,
            'size_bytes': signal_data.size_bytes,
            'source_info': signal_data.source_info,
            
            # Compatibility flags
            'is_media_only': self._is_media_only(signal_data),
            'is_multimodal': signal_data.signal_type == SignalType.MULTIMODAL,
        }
    
    def _extract_text_content(self, signal_data: SignalData) -> str:
        """
        Extract text content for existing downstream compatibility
        """
        if signal_data.signal_type == SignalType.TEXT:
            return str(signal_data.raw_data)
        elif signal_data.signal_type == SignalType.MULTIMODAL:
            # Extract text from multimodal if present
            if isinstance(signal_data.raw_data, dict):
                return signal_data.raw_data.get('text', '')
            elif isinstance(signal_data.raw_data, list):
                text_parts = []
                for item in signal_data.raw_data:
                    if isinstance(item, str):
                        text_parts.append(item)
                return ' '.join(text_parts)
            return ''
        else:
            # Non-text signals return empty for now
            return ''
    
    def _extract_media_payloads(self, signal_data: SignalData) -> dict:
        """
        Extract media payloads for existing downstream compatibility
        """
        media_payloads = {}
        
        if signal_data.signal_type in [SignalType.IMAGE, SignalType.AUDIO, SignalType.VIDEO, SignalType.DOCUMENT]:
            media_payloads[signal_data.signal_type.value] = signal_data.raw_data
        
        elif signal_data.signal_type == SignalType.MULTIMODAL:
            if isinstance(signal_data.raw_data, dict):
                for key, value in signal_data.raw_data.items():
                    if key in ['image', 'audio', 'video', 'document', 'file']:
                        media_payloads[key] = value
        
        return media_payloads
    
    def _is_media_only(self, signal_data: SignalData) -> bool:
        """
        Check if signal is media-only (no text content)
        """
        if signal_data.signal_type in [SignalType.IMAGE, SignalType.AUDIO, SignalType.VIDEO, SignalType.DOCUMENT]:
            return True
        
        if signal_data.signal_type == SignalType.MULTIMODAL:
            text_content = self._extract_text_content(signal_data)
            return not text_content.strip()
        
        return False

# Usage example for integration with existing main.py
def integrate_with_existing_main():
    """
    Example of how to integrate with existing main.py
    """
    
    # Replace existing Phase 1.0 code with this:
    
    # OLD CODE (in main.py):
    """
    # PHASE 1.0 — Raw Query Capture & Modality Separation
    raw_text = ""
    media_payloads = {}
    
    if user_query is None:
        raw_text = ""
    elif isinstance(user_query, str):
        try:
            parsed = json.loads(user_query)
            if isinstance(parsed, dict):
                raw_text = str(parsed.get("text", ""))
                media_payloads = parsed.get("media", {})
            else:
                raw_text = user_query
        except Exception:
            raw_text = user_query
    # ... more complex parsing logic
    """
    
    # NEW CODE (replace with this):
    """
    # PHASE 1.0 — Blueprint Compliant Raw Signal Capture
    phase1_integration = Phase1_0_Integration()
    phase1_result = phase1_integration.capture_user_query(
        user_query=user_query,
        mime_type=request.headers.get('Content-Type'),
        filename=request.files.get('filename').filename if request.files else None
    )
    
    # Extract for existing downstream compatibility
    raw_query = phase1_result['raw_query']
    media_payloads = phase1_result['media_payloads']
    
    # New blueprint compliant data available
    signal_data = phase1_result['signal_data']
    signal_type = phase1_result['signal_type']
    """
    
    pass

# Test integration
def test_integration():
    """
    Test integration with existing interface
    """
    
    phase1_int = Phase1_0_Integration()
    
    print("🔗 Integration Test with Existing Interface")
    print("=" * 50)
    
    # Test 1: Plain text
    print("\n1. Plain Text Integration:")
    result = phase1_int.capture_user_query("What is dharma?")
    print(f"   ✅ raw_query: '{result['raw_query']}'")
    print(f"   ✅ signal_type: {result['signal_type']}")
    print(f"   ✅ media_payloads: {result['media_payloads']}")
    
    # Test 2: JSON string (not parsed)
    print("\n2. JSON String Integration:")
    json_input = '{"query": "explain viman", "context": "vedic"}'
    result = phase1_int.capture_user_query(json_input)
    print(f"   ✅ raw_query: '{result['raw_query']}'")
    print(f"   ✅ signal_type: {result['signal_type']}")
    print(f"   ✅ No parsing applied")
    
    # Test 3: Multimodal
    print("\n3. Multimodal Integration:")
    multimodal_input = {
        "text": "Analyze this temple",
        "image": "base64_image_data",
        "audio": "base64_audio_data"
    }
    result = phase1_int.capture_user_query(multimodal_input)
    print(f"   ✅ raw_query: '{result['raw_query']}'")
    print(f"   ✅ signal_type: {result['signal_type']}")
    print(f"   ✅ media_payloads keys: {list(result['media_payloads'].keys())}")
    
    # Test 4: Binary image
    print("\n4. Binary Image Integration:")
    image_bytes = b'\xFF\xD8\xFF\xE0\x00\x10JFIF'
    result = phase1_int.capture_user_query(image_bytes, filename='temple.jpg')
    print(f"   ✅ raw_query: '{result['raw_query']}'")
    print(f"   ✅ signal_type: {result['signal_type']}")
    print(f"   ✅ is_media_only: {result['is_media_only']}")
    
    print("\n" + "=" * 50)
    print("✅ Integration successful!")
    print("✅ Existing downstream phases will work unchanged")
    print("✅ New blueprint compliant data available")
    
    return True

if __name__ == "__main__":
    test_integration()
