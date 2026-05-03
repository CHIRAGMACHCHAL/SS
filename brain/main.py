# brain/main.py
import os
import logging
from pathlib import Path
import uuid
import csv
# from llm.llm_engine import llm, generate as llm_generate  # Commented out for testing
from memory.conversation_memory import ConversationMemory
from memory.graph_sync import MemoryGraph   
from tools.tools import ToolAgencyLayer   
from memory.vector_store import VectorStoreManager
from memory.ingestion import (
    get_qdrant_client,
    Document,
    EMBEDDING_MODEL,
    PUBLIC_COLLECTION,
    JARVIS_COLLECTION
)
import ftfy
from langdetect import detect as lang_detect
import json
import re
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
import sys
import hashlib
from typing import Dict, Any, Union, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from billing.billing import BillingLayer

class SignalType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    CODE = "code" # Programming languages, scripts
    DATA = "data" # CSV, JSON, structured data
    VOICE = "voice" # For voice interaction
    BINARY = "binary"
    MULTIMODAL = "multimodal"

class QueryFormat(Enum):
    PLAIN_TEXT = "plain_text"
    JSON_API = "json_api"
    MULTIPART = "multipart"
    FILE_UPLOAD = "file_upload"
    BINARY_DATA = "binary_data"
    VOICE_INPUT = "voice_input"
    VISION_INPUT = "vision_input"
    FORM_DATA = "form_data"

# Industry-Standard Magic Byte Map (Physical Properties Only)
MAGIC_BYTE_SIGNATURES = {
    b'\xff\xd8\xff': (SignalType.IMAGE, 'image/jpeg'),
    b'\x89PNG\r\n\x1a\n': (SignalType.IMAGE, 'image/png'),
    b'GIF87a': (SignalType.IMAGE, 'image/gif'),
    b'GIF89a': (SignalType.IMAGE, 'image/gif'),
    b'%PDF': (SignalType.DOCUMENT, 'application/pdf'),
    b'PK\x03\x04': (SignalType.DOCUMENT, 'application/zip'), # DOCX/XLSX/PPTX base
    b'ID3': (SignalType.AUDIO, 'audio/mpeg'),
    b'\xff\xfb': (SignalType.AUDIO, 'audio/mpeg'),
    b'\x00\x00\x00\x18ftyp': (SignalType.VIDEO, 'video/mp4'),
    b'\x1a\x45\xdf\xa3': (SignalType.VIDEO, 'video/webm'),
    # b'RIFF': (SignalType.AUDIO, 'audio/wav'), # Fallback for WAV
    b'#!/': (SignalType.CODE, 'text/x-script'),
}

@dataclass
class SignalData:
    """Immutable Raw Signal Bundle — Passed to Phase 1.5 for Enforcement"""
    capture_id: str
    user_email: str
    signal_type: SignalType
    raw_data: Any
    source_format: str
    content_type: str
    size_bytes: int
    capture_timestamp: str
    metadata: Dict[str, Any]
    tier_limits: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)
    query_format: QueryFormat = QueryFormat.PLAIN_TEXT
    files_meta: Dict[str, Any] = field(default_factory=dict)
    
    # 🔴 EXPLICIT BACKWARD COMPATIBILITY FIELDS (Downstream safe)
    session_id: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None
    platform_info: Optional[Dict[str, Any]] = None
    source_info: Optional[Dict[str, Any]] = None
    language_info: Optional[Dict[str, Any]] = None
    modality_info: Optional[Dict[str, Any]] = None

class Phase1_0_SignalCaptureEngine:
    """
    Pure Capture Layer. No semantic analysis. No blocking.
    Industry: Robust modality detection + tier context injection.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.billing = BillingLayer()

    def capture(
        self,
        raw_input: Any,
        user_email: str,
        content_type: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
        files: Optional[List[Dict[str, Any]]] = None  # ← Added for file uploads
    ) -> SignalData:
        """Main capture entry point — Blueprint: Zero interpretation"""
        try:
            
            capture_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()

            # 1. Detect Modality (Physical/External signals only)
            signal_type, source_format = self._detect_modality(raw_input, content_type, headers)

            # 2. Calculate Exact Byte Size
            size_bytes = self._calculate_size(raw_input)

            # 3. Query Format Detection (External signals only)
            query_format = self._detect_query_format(content_type, bool(files), isinstance(raw_input, bytes))

            # 4. Files Metadata (Count & size only — zero interpretation)
            files_meta = {}
            if files:
                files_meta = {
                    "file_count": len(files),
                    "total_files_size": sum(f.get("size", 0) for f in files),
                    "first_file_type": files[0].get("content_type", "unknown") 
                }

            # 5. Fetch Tier Context (For Phase 1.5 to enforce later)
            try:
                tier_limits = self.billing.get_modality_limits(user_email)
            except Exception:
                tier_limits = BillingLayer.TIER_MODALITY_LIMITS.get('free', {})


            # 6. Build Rich Industry Metadata (Preserved from PerfectPhase1_0)
            metadata = self._build_rich_metadata(headers, session_id, source_format, size_bytes, signal_type, files)

            # Extract explicit fields for direct downstream access
            _session_id = session_id or (headers.get("x-session-id") if headers else None)
            _device_info = metadata.get("device_info")
            _platform_info = metadata.get("platform_info")
            _source_info = metadata.get("source_info")
            _language_info = metadata.get("language_info")
            _modality_info = metadata.get("modality_info")


            signal = SignalData(
                capture_id=capture_id,
                user_email=user_email,
                signal_type=signal_type,
                raw_data=raw_input, # EXACT AS-IS
                source_format=source_format,
                content_type=content_type or "text/plain",
                size_bytes=size_bytes,
                capture_timestamp=timestamp,
                metadata=metadata,
                tier_limits=tier_limits,
                validation_status="pending",
                query_format=query_format,         # ← Added for query format detection
                files_meta=files_meta,
                session_id=_session_id,
                device_info=_device_info,
                platform_info=_platform_info,
                source_info=_source_info,
                language_info=_language_info,
                modality_info=_modality_info 
            )
            return signal
        except Exception as e:
            self.logger.error(f"[Phase1.0] Capture failed safely: {e}")
            # 🔴 FIX: Production Fallback (Blueprint: System tootna nahi chahiye)
            return SignalData(
                capture_id=str(uuid.uuid4()),
                user_email=user_email or "unknown",
                signal_type=SignalType.TEXT,
                raw_data=str(raw_input),
                source_format="fallback/plain",
                content_type="text/plain",
                size_bytes=len(str(raw_input).encode("utf-8")) if raw_input else 0,
                capture_timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"error": str(e), "fallback_mode": True},
                tier_limits=BillingLayer.TIER_MODALITY_LIMITS.get("free", {}),
                validation_status="failed",
                query_format=QueryFormat.PLAIN_TEXT
            )


    def _detect_modality(self, data: Any, ct_header: Optional[str], headers: Optional[Dict]) -> Tuple[SignalType, str]:
        """Detect modality using Content-Type → Magic Bytes → Type Fallback"""
        h = headers or {}
    
        # Priority 0: Explicit voice stream header
        if h.get("x-stream-type", "").lower() == "voice":
            return SignalType.VOICE, "audio/webm;codecs=opus"
        
        # Priority 1: Trusted External Header
        if ct_header:
            clean_ct = ct_header.lower().split(';')[0].strip()
            # Check full ct_header (not split) for opus codec indicator
            signal_type = self._map_content_type(ct_header.lower())
            return signal_type, clean_ct
    

        # Priority 2: Magic Bytes (Physical Signature)
        if isinstance(data, bytes) and len(data) >= 8:
            for sig, (sig_type, mime) in MAGIC_BYTE_SIGNATURES.items():
                if data.startswith(sig):
                    return sig_type, mime

        # WAV special — RIFF is shared with AVI, must verify bytes 8-12
        if isinstance(data, bytes) and data.startswith(b'RIFF') and len(data) > 12 and data[8:12] == b'WAVE':
            return SignalType.AUDIO, 'audio/wav'           

        # Priority 3: Structural Fallback (Zero Interpretation)
        if isinstance(data, str):
            # if data.strip().startswith(('def ', 'class ', 'import ', 'from ', 'function ', 'const ')):
            #     return SignalType.CODE, 'text/plain'
            CODE_INDICATORS = (
                # Python
                'def ', 'class ', 'import ', 'from ',
                # JS/TS
                'function ', 'const ', 'let ', 'var ', 'async ',
                # Systems
                '#include', 'fn ', 'package ', 'pub fn',
                # Shell
                '#!/', 'echo ',
                # Java/C#
                'public class', 'private ', 'void ',
                # Go
                'func ', 'package main',
                # Haskell/Elixir
                'module ', 'defmodule ',
            )
            stripped = data.strip()
            if any(stripped.startswith(indicator) for indicator in CODE_INDICATORS):
                return SignalType.CODE, 'text/plain'
            #---------------------------
            return SignalType.TEXT, 'text/plain'
        elif isinstance(data, (dict, list)):
            return SignalType.DATA, 'application/json'
        elif isinstance(data, bytes):
            return SignalType.BINARY, 'application/octet-stream'
        
        return SignalType.TEXT, 'text/plain'

    def _map_content_type(self, ct: str) -> SignalType:   
        """Map standard MIME to internal SignalType"""
        if ct.startswith('text/'): return SignalType.TEXT
        if ct.startswith('image/'): return SignalType.IMAGE
        if ct.startswith('video/'): return SignalType.VIDEO
        # VOICE vs AUDIO distinction — industry level
        if ct.startswith('audio/'):
            # Real-time voice codecs — Opus is the standard for WebRTC live voice
            if 'opus' in ct or 'webm' in ct:
                return SignalType.VOICE
            return SignalType.AUDIO
        if ct in ['application/pdf', 'application/msword', 'application/zip', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document','application/vnd.ms-excel','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            return SignalType.DOCUMENT
        if ct in ['text/x-python', 'text/javascript', 'text/x-c++', 'application/json']:
            return SignalType.CODE
        return SignalType.BINARY

    def _calculate_size(self, data: Any) -> int:
        """Exact byte size without loading into memory"""
        if isinstance(data, str):
            return len(data.encode('utf-8'))
        if isinstance(data, bytes):
            return len(data)
        if isinstance(data, (dict, list)):
            # Safe approximate for JSON/structured payloads
            try:
                return len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
            except (TypeError, ValueError):
                return len(repr(data).encode('utf-8'))  # safe fallback only
  
        return sys.getsizeof(data)
    
    def _detect_query_format(self, content_type: Optional[str], has_files: bool, is_bytes: bool) -> QueryFormat:
        # if has_files: return QueryFormat.MULTIPART
        if has_files:
            return QueryFormat.FILE_UPLOAD  # dedicated file upload
        if content_type and content_type.startswith("multipart/"):
            return QueryFormat.MULTIPART  # form with files        if content_type:
            if content_type.startswith("application/json"): return QueryFormat.JSON_API
            if content_type.startswith("multipart/"): return QueryFormat.MULTIPART
            if content_type.startswith("application/x-www-form-urlencoded"): return QueryFormat.FORM_DATA
            if content_type.startswith("application/octet-stream"): return QueryFormat.BINARY_DATA
            if content_type.startswith("audio/"): return QueryFormat.VOICE_INPUT
            if content_type.startswith(("image/", "video/")): return QueryFormat.VISION_INPUT
        if is_bytes: return QueryFormat.BINARY_DATA
        return QueryFormat.PLAIN_TEXT



    def _build_rich_metadata(self, headers: Optional[Dict], session_id: Optional[str], source_fmt: str, size: int, files: Optional[List] = None) -> Dict[str, Any]:
        """Restores PerfectPhase1_0's rich telemetry while staying blueprint-compliant"""
        return {
            "session_id": session_id or str(uuid.uuid4()),
            "client_ip": headers.get("x-forwarded-for", "unknown") if headers else "unknown",
            "device_info": {
                "user_agent": headers.get("user-agent") if headers else None,
                "platform": headers.get("x-platform") if headers else None,
                "app_version": headers.get("x-app-version") if headers else None,
                "device_type": headers.get("x-device-type") if headers else None
            },
            "platform_info": {
                "os": headers.get("x-os") if headers else None,
                "browser": headers.get("x-browser") if headers else None,
                "screen_resolution": headers.get("x-screen-resolution") if headers else None,
                "timezone": headers.get("x-timezone") if headers else None
            },
            "source_info": {
                "has_files": bool(files) and len(files) > 0,
                "file_count": len(files) if files else 0,
                "content_type_provided": bool(headers and "content-type" in headers),
                "session_provided": bool(headers and "x-session-id" in headers)
            },
            "language_info": {
                "preferred_language": headers.get("accept-language") if headers else None,
                "content_language": headers.get("content-language") if headers else None,
                "region": headers.get("x-region") if headers else None,
                "locale": headers.get("x-locale") if headers else None
            },
            "modality_info": {
                # "has_voice": any(f.get("content_type", "").startswith("audio/") for f in files) if files else False,
                "has_voice": (signal_type == SignalType.VOICE) or (bool(files) and any(f.get("content_type","").startswith("audio/") for f in files)),

                # "has_vision": any(f.get("content_type", "").startswith(("image/", "video/")) for f in files) if files else False,
                "has_vision": (signal_type in (SignalType.IMAGE, SignalType.VIDEO)) or (bool(files) and any(f.get("content_type","").startswith(("image/","video/")) for f in files)),
                # "has_text": isinstance(headers.get("content-type"), str) and headers["content-type"].startswith("text/") if headers else True,
                "has_text": isinstance(raw_data, str) and len(raw_data.strip()) > 0 if raw_data is not None else False,
                "has_multimodal": bool(files) and len(files) > 0
            },
            "original_format": source_fmt,
            "capture_method": "phase1_0_raw",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload_size_bytes": size,

        }

def create_signal_capture_engine() -> Phase1_0_SignalCaptureEngine:
    """Factory — Single instance, production ready"""
    return Phase1_0_SignalCaptureEngine()

# 1. Initialize Engine
signal_engine = Phase1_0_SignalCaptureEngine()  

#----------------
_signal_engine: Optional[Phase1_0_SignalCaptureEngine] = None

def get_signal_engine() -> Phase1_0_SignalCaptureEngine:
    global _signal_engine
    if _signal_engine is None:
        _signal_engine = Phase1_0_SignalCaptureEngine()
    return _signal_engine


# ════════════════════════════════════════════════════════════════
# PHASE 1.1 — MULTIMODAL NORMALIZATION ENGINE
# Blueprint: "Har modality ko processable canonical form mein convert karna,
#             bina intent distort kiye"
# Industry:  Real processing — Pillow, librosa, pypdf, pytesseract, ast
#            Language-agnostic — spaCy xx_ent_wiki_sm (50+ languages)
#            Tier-aware — config ke modalities list se, size se nahi
# ════════════════════════════════════════════════════════════════



@dataclass
class NormalizedSignal:
    """
    Phase 1.1 ka output. Phase 1.2 aur aage yahi use karta hai.
    canonical_text: downstream ke liye processable text form
    media_metadata: image/audio/video se extracted signals
    linguistic_signals: spaCy se — language agnostic
    """
    modality          : str
    canonical_text    : str                    # har modality ka text form
    raw_data          : Any                    # original as-is
    language          : str                    # langdetect result
    linguistic_signals: Dict[str, Any]         # spaCy signals
    media_metadata    : Dict[str, Any]         # image/audio/video info
    processing_steps  : List[str]              # kya kya hua
    success           : bool        = True
    error             : Optional[str] = None


class Phase1_1_NormalizationEngine:
    """
    Blueprint: "normalize, noise hatao, grammar perfect karne ki koshish karo"
    Ek hi class — sab modalities handle karta hai.
    Tier-aware: tier_limits['modalities'] list se check — size se nahi.
    Language-agnostic: xx_ent_wiki_sm multilingual spaCy model.
    No placeholders — real libraries, real processing.
    """

    # Standard image size for normalization
    _IMG_STD_SIZE = (1024, 1024)
    # Audio sample rate standard
    _AUDIO_SR     = 16000
    # Whisper model — load lazily
    _whisper_model = None

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # ── Public entry point ────────────────────────────────────────────────

    def normalize(
        self,
        raw_data    : Any,
        signal_type : str,
        tier_limits : Dict[str, Any],
        content_type: str = "text/plain",
    ) -> NormalizedSignal:
        """
        Main entry. raw_data + signal_type → NormalizedSignal.
        Tier check: kya yeh modality is tier ko allowed hai?
        """
        allowed = tier_limits.get("modalities", ["text"])
        if signal_type not in allowed:
            return NormalizedSignal(
                modality           = signal_type,
                canonical_text     = "",
                raw_data           = raw_data,
                language           = "unknown",
                linguistic_signals = {},
                media_metadata     = {},
                processing_steps   = [],
                success            = False,
                error              = f"Modality '{signal_type}' not allowed at this tier",
            )

        dispatch = {
            "text"      : self._normalize_text,
            "code"      : self._normalize_code,
            "data"      : self._normalize_data,
            "image"     : self._normalize_image,
            "audio"     : self._normalize_audio,
            "voice"     : self._normalize_voice,
            "video"     : self._normalize_video,
            "document"  : self._normalize_document,
            "binary"    : self._normalize_binary,
            "multimodal": self._normalize_multimodal,
        }

        handler = dispatch.get(signal_type, self._normalize_binary)
        try:
            return handler(raw_data, tier_limits)
        except Exception as e:
            self.logger.error(f"[Ph1.1] {signal_type} normalization failed: {e}")
            return NormalizedSignal(
                modality           = signal_type,
                canonical_text     = str(raw_data)[:500] if raw_data else "",
                raw_data           = raw_data,
                language           = "unknown",
                linguistic_signals = {},
                media_metadata     = {"error": str(e)},
                processing_steps   = ["fallback"],
                success            = False,
                error              = str(e),
            )

    # ── TEXT ─────────────────────────────────────────────────────────────

    def _normalize_text(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Blueprint: "Unicode fix (ftfy), whitespace cleanup, language detection"
        spaCy xx_ent_wiki_sm — multilingual, language-agnostic NER + POS
        """
        steps = []

        # 1. Ensure string
        text = raw_data if isinstance(raw_data, str) else str(raw_data)

        # 2. Unicode fix — mojibake, broken encoding sab theek karo
        text = ftfy.fix_text(text)
        steps.append("ftfy_unicode_fix")

        # Global repetition normalize — language agnostic Unicode-aware
        # "!!!!" → "!!", "हाँ!!!" → "हाँ!!", koi bhi script
        import re as _re
        text = _re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # ALL CAPS normalize — sirf agar poora text upper hai (emotional shouting)
        # unicodedata se check — script-agnostic
        if text.upper() == text and len(text.split()) > 2:
            text = text[0].upper() + text[1:].lower()

        # 3. Whitespace normalize — tabs, multiple spaces, newlines
        text = re.sub(r'\s+', ' ', text.strip())
        steps.append("whitespace_normalize")

        # 4. Language detection — 50+ languages, Hindi bhi Sanskrit bhi
        try:
            lang = lang_detect(text)
        except Exception:
            lang = "unknown"
        steps.append("langdetect")

        # 5. spaCy — xx_ent_wiki_sm is multilingual
        #    dep_/pos_/ents — language agnostic tags
        doc    = _NLP(text[:2000])
        ents   = [(e.text, e.label_) for e in doc.ents]
        chunks = [c.text for c in doc.noun_chunks]

        signals = {
            "entity_count"   : len(ents),
            "entities"       : ents,
            "noun_chunks"    : chunks,
            "verb_count"     : sum(1 for t in doc if t.pos_ == "VERB"),
            "noun_count"     : sum(1 for t in doc if t.pos_ == "NOUN"),
            "sentence_count" : sum(1 for _ in doc.sents),
            "has_causal"     : any(
                t.dep_ in ("advcl", "mark", "csubj", "relcl")
                for t in doc
            ),
            "is_analytical"  : any(
                t.dep_ in ("advcl", "ccomp", "expl")
                for t in doc
            ),
            "has_entity"     : len(ents) >= 1,
            "has_verb"       : any(t.pos_ == "VERB" for t in doc),
        }
        steps.append("spacy_multilingual_signals")

        return NormalizedSignal(
            modality           = "text",
            canonical_text     = text,
            raw_data           = raw_data,
            language           = lang,
            linguistic_signals = signals,
            media_metadata     = {},
            processing_steps   = steps,
        )

    # ── IMAGE ─────────────────────────────────────────────────────────────

    def _normalize_image(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Blueprint: "resolution normalization, format standardization,
                    noise reduction, optional OCR"
        Pillow: resize + RGB normalize + format detect
        pytesseract: OCR — Devanagari/Sanskrit bhi (tier-aware)
        """
        steps  = []
        meta   = {}

        try:
            from PIL import Image as PILImage
            import io as _io

            # Decode bytes → PIL Image
            if isinstance(raw_data, bytes):
                img = PILImage.open(_io.BytesIO(raw_data))
            elif isinstance(raw_data, str):
                import base64 as _b64
                img = PILImage.open(_io.BytesIO(_b64.b64decode(raw_data + "==")))
            else:
                raise ValueError(f"Cannot decode image from {type(raw_data)}")

            meta["original_format"] = img.format or "unknown"
            meta["original_size"]   = list(img.size)
            meta["original_mode"]   = img.mode
            steps.append("image_decode")

            # Convert to RGB — removes alpha, standardizes channels
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
                steps.append("rgb_convert")

            # Resize to standard — LANCZOS is best quality downscale
            if img.size[0] > self._IMG_STD_SIZE[0] or img.size[1] > self._IMG_STD_SIZE[1]:
                img.thumbnail(self._IMG_STD_SIZE, PILImage.LANCZOS)
                steps.append("resize_standard")

            meta["normalized_size"] = list(img.size)

            # Dominant color extraction — numpy mean per channel
            import numpy as _np
            arr               = _np.array(img)
            meta["mean_rgb"]  = [round(float(arr[:,:,c].mean()), 2)
                                  for c in range(arr.shape[2])]
            meta["std_rgb"]   = [round(float(arr[:,:,c].std()), 2)
                                  for c in range(arr.shape[2])]
            steps.append("color_analysis")

            # OCR — tier-aware: business_small+ only
            # pytesseract supports 100+ languages including Devanagari
            ocr_text = ""
            ocr_allowed = set(["business_small", "enterprise", "jarvis"])
            # Check via max_files as proxy (business_small = 50+)
            max_files_val = tier_limits.get("max_files", 3)
            if tier_limits.get("ocr_enabled", False):
            # if max_files_val == -1 or max_files_val >= 50:
                try:
                    import pytesseract
                    # lang='hin+eng+san' — Hindi + English + Sanskrit
                    ocr_text = pytesseract.image_to_string(
                        img, lang="hin+eng+san"
                    ).strip()
                    meta["ocr_text"]   = ocr_text
                    meta["ocr_length"] = len(ocr_text)
                    steps.append("pytesseract_ocr_multilingual")
                except ImportError:
                    self.logger.warning("[Ph1.1 IMAGE] pytesseract not installed")
                except Exception as e:
                    self.logger.warning(f"[Ph1.1 IMAGE] OCR failed: {e}")

            # canonical_text: OCR text agar mila, warna image description
            w, h   = img.size
            aspect = round(w / max(h, 1), 3)
            canonical = (
                f"IMAGE w={w} h={h} aspect={aspect} "
                f"format={meta['original_format']} mode=RGB"
            )
            if ocr_text:
                canonical += f" | OCR: {ocr_text[:300]}"

            # Run text signals on canonical
            ling = self._text_signals(canonical)

            return NormalizedSignal(
                modality           = "image",
                canonical_text     = canonical,
                raw_data           = raw_data,
                language           = ling["language"],
                linguistic_signals = ling["signals"],
                media_metadata     = meta,
                processing_steps   = steps,
            )

        except ImportError:
            return self._pillow_missing_fallback(raw_data, steps)

    def _pillow_missing_fallback(self, raw_data, steps):
        size = len(raw_data) if isinstance(raw_data, bytes) else 0
        return NormalizedSignal(
            modality           = "image",
            canonical_text     = f"IMAGE size={size}bytes (Pillow not installed)",
            raw_data           = raw_data,
            language           = "unknown",
            linguistic_signals = {},
            media_metadata     = {"error": "Pillow not installed", "size_bytes": size},
            processing_steps   = steps + ["fallback_no_pillow"],
            success            = False,
            error              = "Pillow not installed",
        )

    # ── AUDIO ─────────────────────────────────────────────────────────────

    def _normalize_audio(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Blueprint: "speech-to-text (ASR), noise filtering, speaker clarity"
        librosa: waveform load + RMS normalize + Log-Mel spectrogram stats
        Whisper: multilingual ASR — Sanskrit bhi detect karta hai (tier-aware)
        """
        steps = []
        meta  = {}

        try:
            import librosa as _librosa
            import io as _io
            import numpy as _np

            # Decode bytes → waveform
            if isinstance(raw_data, bytes):
                buf = _io.BytesIO(raw_data)
            else:
                raise ValueError("Audio must be bytes")

            # librosa load — mono, 16kHz standard
            y, sr = _librosa.load(buf, sr=self._AUDIO_SR, mono=True)
            steps.append("librosa_load_16khz_mono")

            meta["duration_sec"]   = round(float(len(y) / sr), 2)
            meta["sample_rate"]    = sr
            meta["sample_count"]   = len(y)

            # RMS normalize — volume standardize karo
            rms = _np.sqrt(_np.mean(y ** 2))
            if rms > 0:
                y = y / rms
            steps.append("rms_normalize")
            meta["rms_before"]     = round(float(rms), 6)

            # Log-Mel Spectrogram — 80 mel bands, Whisper-compatible
            # Tone, pitch, Sanskrit shlok ka uchcharan — sab preserve
            mel = _librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=80, hop_length=160, n_fft=400
            )
            log_mel = _librosa.power_to_db(mel, ref=_np.max)
            meta["mel_shape"]      = list(log_mel.shape)
            meta["mel_mean"]       = round(float(log_mel.mean()), 4)
            meta["mel_std"]        = round(float(log_mel.std()), 4)

            # Spectral centroid — brightness of audio
            centroid = _librosa.feature.spectral_centroid(y=y, sr=sr)
            meta["spectral_centroid_mean"] = round(float(centroid.mean()), 2)
            steps.append("log_mel_spectrogram")

            # Zero Crossing Rate — roughness/noisiness
            zcr = _librosa.feature.zero_crossing_rate(y)
            meta["zcr_mean"]       = round(float(zcr.mean()), 4)

            # Tempo estimate
            try:
                tempo, _ = _librosa.beat.beat_track(y=y, sr=sr)
                meta["tempo_bpm"]  = round(float(tempo), 1)
            except Exception:
                meta["tempo_bpm"]  = None

            steps.append("acoustic_features")

            # ASR — Whisper multilingual (paid+ tier)
            transcript = ""
            if tier_limits.get("asr_enabled", False):
            # if tier_limits.get("audio_limit_mb", 0) >= 50:
                transcript = self._whisper_transcribe(raw_data)
                if transcript:
                    meta["transcript"]         = transcript
                    meta["transcript_length"]  = len(transcript)
                    steps.append("whisper_multilingual_asr")

            canonical = (
                f"AUDIO duration={meta['duration_sec']}s "
                f"sr={sr}Hz mel_mean={meta['mel_mean']} "
                f"centroid={meta['spectral_centroid_mean']}Hz"
            )
            if transcript:
                canonical += f" | TRANSCRIPT: {transcript[:400]}"

            ling = self._text_signals(canonical)

            return NormalizedSignal(
                modality           = "audio",
                canonical_text     = canonical,
                raw_data           = raw_data,
                language           = ling["language"],
                linguistic_signals = ling["signals"],
                media_metadata     = meta,
                processing_steps   = steps,
            )

        except ImportError:
            return self._librosa_missing_fallback("audio", raw_data, steps)

    # ── VOICE ─────────────────────────────────────────────────────────────

    def _normalize_voice(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Voice = live stream — WebRTC Opus chunks.
        Same processing as audio lekin chunked.
        Pitch + tone preservation critical (Vedic mantra uccharan).
        """
        steps = ["voice_stream_detected"]
        meta  = {}

        try:
            import librosa as _librosa
            import io as _io
            import numpy as _np

            buf = _io.BytesIO(raw_data) if isinstance(raw_data, bytes) else raw_data
            y, sr = _librosa.load(buf, sr=self._AUDIO_SR, mono=True)
            steps.append("librosa_load_voice_chunk")

            # RMS
            rms = _np.sqrt(_np.mean(y ** 2))
            if rms > 0:
                y = y / rms
            meta["rms"] = round(float(rms), 6)

            # Mel — smaller window for real-time
            mel = _librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
            log_mel = _librosa.power_to_db(mel, ref=_np.max)
            meta["mel_mean"] = round(float(log_mel.mean()), 4)
            meta["duration_sec"] = round(float(len(y) / sr), 3)
            steps.append("voice_mel_spectrogram")

            # Pitch detection — F0 fundamental frequency
            try:
                f0, _, _ = _librosa.pyin(
                    y, fmin=_librosa.note_to_hz('C2'),
                    fmax=_librosa.note_to_hz('C7')
                )
                valid_f0 = f0[~_np.isnan(f0)] if f0 is not None else []
                meta["pitch_mean_hz"] = round(float(_np.mean(valid_f0)), 2) if len(valid_f0) > 0 else 0.0
                steps.append("pitch_detection_f0")
            except Exception:
                meta["pitch_mean_hz"] = 0.0

            # ASR for voice (ultra_paid+)
            transcript = ""
            if tier_limits.get("voice_limit_mb", 0) >= 50:
                transcript = self._whisper_transcribe(raw_data)
                if transcript:
                    meta["transcript"] = transcript
                    steps.append("whisper_voice_asr")

            canonical = (
                f"VOICE chunk={meta['duration_sec']}s "
                f"pitch={meta['pitch_mean_hz']}Hz "
                f"rms={meta['rms']}"
            )
            if transcript:
                canonical += f" | LIVE: {transcript[:200]}"

            ling = self._text_signals(canonical)

            return NormalizedSignal(
                modality           = "voice",
                canonical_text     = canonical,
                raw_data           = raw_data,
                language           = ling["language"],
                linguistic_signals = ling["signals"],
                media_metadata     = meta,
                processing_steps   = steps,
            )

        except ImportError:
            return self._librosa_missing_fallback("voice", raw_data, steps)

    # ── VIDEO ─────────────────────────────────────────────────────────────

    def _normalize_video(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Blueprint: "keyframe extraction, audio separation, frame-level signals"
        OpenCV: frame extraction + motion detection
        Tier-aware: business_small+ only gets video
        """
        steps = []
        meta  = {}

        try:
            import cv2 as _cv2
            import tempfile as _tmp
            import os as _os
            import numpy as _np

            # Write to temp file — OpenCV needs file path
            with _tmp.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(raw_data if isinstance(raw_data, bytes) else b"")
                tmp_path = f.name

            try:
                cap = _cv2.VideoCapture(tmp_path)
                # ... (rest of the video processing code) ...
            finally:
                # GUARANTEE: File will be deleted even if OpenCV crashes
                if '_os' in locals() or 'os' in locals():
                    import os as _os
                    if _os.path.exists(tmp_path):
                        _os.unlink(tmp_path)

            meta["fps"]          = round(cap.get(_cv2.CAP_PROP_FPS), 2)
            meta["frame_count"]  = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            meta["width"]        = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            meta["height"]       = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            meta["duration_sec"] = round(
                meta["frame_count"] / max(meta["fps"], 1), 2
            )
            steps.append("cv2_video_open")

            # Keyframe extraction — every N frames
            n_keyframes   = 5
            keyframe_interval = max(1, meta["frame_count"] // n_keyframes)
            keyframes     = []
            frame_idx     = 0
            prev_gray     = None
            motion_scores = []

            while cap.isOpened() and len(keyframes) < n_keyframes:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % keyframe_interval == 0:
                    gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
                    # Motion score vs previous keyframe
                    if prev_gray is not None:
                        diff = _np.abs(
                            gray.astype(_np.float32) - prev_gray.astype(_np.float32)
                        )
                        motion_scores.append(round(float(diff.mean()), 3))
                    prev_gray = gray
                    keyframes.append(frame_idx)
                frame_idx += 1

            cap.release()
            _os.unlink(tmp_path)

            meta["keyframes_extracted"] = keyframes
            meta["motion_scores"]       = motion_scores
            meta["avg_motion"]          = round(
                float(sum(motion_scores) / len(motion_scores)), 3
            ) if motion_scores else 0.0
            steps.append("keyframe_extraction")
            steps.append("motion_detection")

            canonical = (
                f"VIDEO duration={meta['duration_sec']}s "
                f"fps={meta['fps']} "
                f"res={meta['width']}x{meta['height']} "
                f"keyframes={len(keyframes)} "
                f"avg_motion={meta['avg_motion']}"
            )

            ling = self._text_signals(canonical)

            return NormalizedSignal(
                modality           = "video",
                canonical_text     = canonical,
                raw_data           = raw_data,
                language           = "unknown",
                linguistic_signals = ling["signals"],
                media_metadata     = meta,
                processing_steps   = steps,
            )

        except ImportError:
            size = len(raw_data) if isinstance(raw_data, bytes) else 0
            return NormalizedSignal(
                modality           = "video",
                canonical_text     = f"VIDEO size={size}bytes (OpenCV not installed)",
                raw_data           = raw_data,
                language           = "unknown",
                linguistic_signals = {},
                media_metadata     = {"error": "OpenCV not installed", "size_bytes": size},
                processing_steps   = steps + ["fallback_no_cv2"],
                success            = False,
                error              = "OpenCV not installed",
            )

    # ── DOCUMENT ──────────────────────────────────────────────────────────

    def _normalize_document(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Blueprint: "text extraction (PDF→text), structure parsing"
        pypdf: real PDF text extract
        Structure: headings, tables detect
        """
        steps = []
        meta  = {}

        try:
            from pypdf import PdfReader
            import io as _io

            buf    = _io.BytesIO(raw_data if isinstance(raw_data, bytes) else str(raw_data).encode())
            reader = PdfReader(buf)
            steps.append("pypdf_open")

            meta["page_count"] = len(reader.pages)
            pages_text         = []

            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text() or ""
                    pages_text.append(t)
                except Exception:
                    pages_text.append("")

            full_text = "\n".join(pages_text).strip()
            steps.append("text_extraction")

            meta["char_count"]  = len(full_text)
            meta["word_count"]  = len(full_text.split())

            # Structure detection — headings (short ALL-CAPS lines)
            lines       = full_text.split("\n")
            headings    = [
                l.strip() for l in lines
                if 3 < len(l.strip()) < 80 and l.strip().isupper()
            ]
            meta["heading_count"]   = len(headings)
            meta["sample_headings"] = headings[:5]
            meta["has_tables"]      = "\t" in full_text or "  |  " in full_text
            steps.append("structure_detection")

            # Normalize extracted text
            norm   = ftfy.fix_text(full_text)
            norm   = re.sub(r'\s+', ' ', norm.strip())

            # Language detect on first 500 chars
            try:
                lang = lang_detect(norm[:500])
            except Exception:
                lang = "unknown"
            steps.append("langdetect")

            ling = self._text_signals(norm[:2000])

            return NormalizedSignal(
                modality           = "document",
                canonical_text     = norm[:4000],   # downstream ke liye reasonable limit
                raw_data           = raw_data,
                language           = lang,
                linguistic_signals = ling["signals"],
                media_metadata     = meta,
                processing_steps   = steps,
            )

        except ImportError:
            return NormalizedSignal(
                modality           = "document",
                canonical_text     = "DOCUMENT (pypdf not installed)",
                raw_data           = raw_data,
                language           = "unknown",
                linguistic_signals = {},
                media_metadata     = {"error": "pypdf not installed"},
                processing_steps   = ["fallback_no_pypdf"],
                success            = False,
                error              = "pypdf not installed",
            )

    # ── CODE ──────────────────────────────────────────────────────────────

    def _normalize_code(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Blueprint: "syntax parsing, structure extraction"
        ast: Python syntax parse — stdlib, no extra install
        Language detect: shebang + keywords — no English string matching on user query
        """
        steps     = []
        meta      = {}

        code_text = (
            raw_data.decode("utf-8", errors="replace")
            if isinstance(raw_data, bytes)
            else str(raw_data)
        )

        # Language detection — Structural & Shebang analysis
        lang = "unknown"
        first_line = code_text.split("\n")[0].lower() if code_text else ""
        
        # 1. Shebang check (Standard)
        if "python" in first_line: lang = "python"
        elif "node" in first_line or "javascript" in first_line: lang = "javascript"
        elif "bash" in first_line or "sh" in first_line: lang = "bash"
        
        # 2. Structural Analysis (Real Power - Language Agnostic)
        if lang == "unknown":
            # Python structural markers
            if "def " in code_text and ("import " in code_text or "class " in code_text):
                lang = "python"
            # JS/TS structural markers
            elif "function " in code_text and ("const " in code_text or "let " in code_text):
                lang = "javascript"
            # Java/C# structural markers
            elif "public class " in code_text and "void main" in code_text:
                lang = "java"
            # C/C++ structural markers
            elif "#include <" in code_text:
                lang = "cpp"
        meta["language"] = lang
        steps.append("language_detection_shebang")

        # Syntax parse — Python only via ast
        meta["valid_syntax"] = None
        if lang == "python":
            try:
                import ast as _ast
                _ast.parse(code_text)
                meta["valid_syntax"] = True
                steps.append("ast_parse_success")
            except SyntaxError as e:
                meta["valid_syntax"] = False
                meta["syntax_error"] = str(e)
                steps.append("ast_parse_failed")

        # Structure metrics
        lines = code_text.split("\n")
        meta["line_count"]      = len(lines)
        meta["function_count"]  = (
            code_text.count("def ") + code_text.count("function ")
        )
        meta["class_count"]     = code_text.count("class ")
        meta["import_count"]    = (
            code_text.count("import ") + code_text.count("from ")
        )
        meta["complexity_score"] = (
            code_text.count("if ")
            + code_text.count("for ")
            + code_text.count("while ")
            + code_text.count("try ")
        )
        steps.append("structure_extraction")

        canonical = (
            f"CODE lang={lang} lines={meta['line_count']} "
            f"functions={meta['function_count']} "
            f"classes={meta['class_count']} "
            f"complexity={meta['complexity_score']}"
        )
        if meta["valid_syntax"] is False:
            canonical += f" SYNTAX_ERROR={meta.get('syntax_error','')}"

        # Code text as context
        canonical += f" | {code_text[:300]}"

        ling = self._text_signals(canonical)

        return NormalizedSignal(
            modality           = "code",
            canonical_text     = canonical,
            raw_data           = raw_data,
            language           = lang,
            linguistic_signals = ling["signals"],
            media_metadata     = meta,
            processing_steps   = steps,
        )

    # ── DATA ──────────────────────────────────────────────────────────────

    def _normalize_data(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Blueprint: "structure extraction"
        stdlib json + csv — no extra install
        """
        steps = []
        meta  = {}

        text = (
            raw_data.decode("utf-8", errors="replace")
            if isinstance(raw_data, bytes)
            else str(raw_data)
        )

        # Try JSON
        try:
            parsed = json.loads(text)
            meta["format"]       = "json"
            meta["keys"]         = list(parsed.keys()) if isinstance(parsed, dict) else []
            meta["record_count"] = len(parsed) if isinstance(parsed, list) else 1
            meta["nested"]       = any(
                isinstance(v, (dict, list)) for v in parsed.values()
            ) if isinstance(parsed, dict) else False
            steps.append("json_parse")
        except (json.JSONDecodeError, Exception):
            # Try CSV
            try:
                import io as _io
                reader = csv.reader(_io.StringIO(text))
                rows   = list(reader)
                meta["format"]        = "csv"
                meta["columns"]       = rows[0] if rows else []
                meta["column_count"]  = len(rows[0]) if rows else 0
                meta["record_count"]  = max(0, len(rows) - 1)
                steps.append("csv_parse")
            except Exception:
                meta["format"] = "plain_text"
                steps.append("plain_text_fallback")

        meta["char_count"] = len(text)
        canonical = (
            f"DATA format={meta['format']} "
            f"records={meta.get('record_count', 0)} "
            f"keys={meta.get('keys', meta.get('columns', []))[:5]}"
        )
        canonical += f" | {text[:200]}"

        ling = self._text_signals(canonical)

        return NormalizedSignal(
            modality           = "data",
            canonical_text     = canonical,
            raw_data           = raw_data,
            language           = ling["language"],
            linguistic_signals = ling["signals"],
            media_metadata     = meta,
            processing_steps   = steps,
        )

    # ── BINARY ────────────────────────────────────────────────────────────

    def _normalize_binary(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """Unknown binary — hash + size. No interpretation."""
        steps = []
        data  = raw_data if isinstance(raw_data, bytes) else str(raw_data).encode()

        # 1. Physical Properties
        sha256    = hashlib.sha256(data).hexdigest()
        size      = len(data)

        # 2. Entropy Calculation (Industry standard for binary analysis)
        # High entropy = encrypted/compressed, Low entropy = structured/sparse
        if size > 0:
            counts = np.bincount(np.frombuffer(data, dtype=np.uint8))
            probs = counts / size
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
        else:
            entropy = 0.0

        canonical = f"BINARY size={size}bytes entropy={round(entropy, 2)} sha256={sha256[:16]}..."
        steps.append("entropy_analysis")

        return NormalizedSignal(
            modality           = "binary",
            canonical_text     = canonical,
            raw_data           = raw_data,
            language           = "unknown",
            linguistic_signals = {},
            media_metadata     = {"size_bytes": size, "sha256": sha256, "entropy": entropy},
            processing_steps   = ["hash_sha256"] + steps,
        )

    # ── MULTIMODAL ────────────────────────────────────────────────────────

    def _normalize_multimodal(self, raw_data: Any, tier_limits: Dict) -> NormalizedSignal:
        """
        Jarvis only — multiple modalities combined.
        raw_data expected as dict: {"text": ..., "image": ..., "audio": ...}
        """
        steps  = ["multimodal_detected"]
        parts  = []
        meta   = {}

        if not isinstance(raw_data, dict):
            return self._normalize_binary(raw_data, tier_limits)

        for mod, data in raw_data.items():
            if mod in ("text", "code", "data", "image", "audio", "video", "document"):
                norm = self.normalize(data, mod, tier_limits)
                if norm.success:
                    parts.append(f"[{mod.upper()}] {norm.canonical_text[:200]}")
                    meta[mod] = norm.media_metadata
                    steps.append(f"sub_normalize_{mod}")

        canonical = " | ".join(parts) if parts else "MULTIMODAL (empty)"
        ling      = self._text_signals(canonical)

        return NormalizedSignal(
            modality           = "multimodal",
            canonical_text     = canonical,
            raw_data           = raw_data,
            language           = ling["language"],
            linguistic_signals = ling["signals"],
            media_metadata     = meta,
            processing_steps   = steps,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _text_signals(self, text: str) -> Dict[str, Any]:
        """spaCy signals on any text — used by media normalizers too."""
        if not text or not text.strip():
            return {"language": "unknown", "signals": {}}
        try:
            lang = lang_detect(text[:300])
        except Exception:
            lang = "unknown"
        doc     = _NLP(text[:1000])
        signals = {
            "entity_count": len(doc.ents),
            "verb_count"  : sum(1 for t in doc if t.pos_ == "VERB"),
            "noun_count"  : sum(1 for t in doc if t.pos_ == "NOUN"),
            "has_entity"  : len(doc.ents) >= 1,
            "has_verb"    : any(t.pos_ == "VERB" for t in doc),
            "is_analytical": any(
                t.dep_ in ("advcl", "ccomp", "expl") for t in doc
            ),
        }
        return {"language": lang, "signals": signals}

    def _librosa_missing_fallback(
        self, modality: str, raw_data: Any, steps: List[str]
    ) -> NormalizedSignal:
        size = len(raw_data) if isinstance(raw_data, bytes) else 0
        return NormalizedSignal(
            modality           = modality,
            canonical_text     = f"{modality.upper()} size={size}bytes (librosa not installed)",
            raw_data           = raw_data,
            language           = "unknown",
            linguistic_signals = {},
            media_metadata     = {"error": "librosa not installed", "size_bytes": size},
            processing_steps   = steps + ["fallback_no_librosa"],
            success            = False,
            error              = "librosa not installed",
        )

    def _whisper_transcribe(self, audio_bytes: bytes) -> str:
        """Whisper ASR — lazy load, multilingual, 99 languages."""
        try:
            import whisper as _whisper
            import io as _io
            import tempfile as _tmp
            import os as _os

            if Phase1_1_NormalizationEngine._whisper_model is None:
                # tiny model — fast, multilingual, reasonable accuracy
                Phase1_1_NormalizationEngine._whisper_model = _whisper.load_model("tiny")

            with _tmp.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp = f.name

            result    = Phase1_1_NormalizationEngine._whisper_model.transcribe(tmp)
            _os.unlink(tmp)
            return result.get("text", "").strip()
        except ImportError:
            self.logger.warning("[Ph1.1] Whisper not installed — ASR skipped")
            return ""
        except Exception as e:
            self.logger.warning(f"[Ph1.1] Whisper failed: {e}")
            return ""


# Module-level singleton
_PHASE1_1 = Phase1_1_NormalizationEngine()

# ════════════════════════════════════
# PHASE 1.01 — Validation Layer (NEW)
# Blueprint: Real enforcement of billing limits
# Industry: Production-grade access control
# ════════════════════════════════════




@dataclass
class ValidationResult:
    """Validation result for enforcement"""
    allowed: bool
    violations: List[str]
    filtered_data: Any
    tier: str
    limits_applied: Dict[str, Any]
class Phase1_5ValidationLayer:
    """
    ENFORCEMENT LAYER - Critical Missing Piece
    Connects Phase 1.0 capture with Billing limits
    Blueprint: Real enforcement, not just config
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    def validate_request(self, 
                        signal_data: SignalData, 
                        user_email: str,
                        files: List[Dict[str, Any]] = None) -> ValidationResult:
        """
        Main validation function - ENFORCES billing limits
        Blueprint: Real power control
        """
        # Step 1: Get user tier and limits
        tier = BillingLayer.get_user_tier(user_email)
        limits = BillingLayer.get_modality_limits(user_email)
        violations = []
        filtered_files = []
        # Step 2: Validate modality access
        modality_violations = self._validate_modalities(signal_data, limits)
        violations.extend(modality_violations)
        # Step 3: Validate files if present
        if files:
            file_violations, filtered_files = self._validate_files(files, limits)
            violations.extend(file_violations)
        # Step 4: Validate content size
        size_violations = self._validate_content_size(signal_data, limits)
        violations.extend(size_violations)
        # Step 5: Final decision
        allowed = len(violations) == 0
        # Step 6: Filter data if needed
        filtered_data = self._filter_signal_data(signal_data, filtered_files) if allowed else None
        return ValidationResult(
            allowed=allowed,
            violations=violations,
            filtered_data=filtered_data,
            tier=tier,
            limits_applied=limits
        )
    def _validate_modalities(self, signal_data: SignalData, limits: Dict[str, Any]) -> List[str]:
        """Validate signal type against tier modalities"""
        violations = []
        signal_type = signal_data.signal_type.value
        allowed_modalities = limits['modalities']
        if signal_type not in allowed_modalities:
            violations.append(f"Modality '{signal_type}' not allowed for tier. Allowed: {allowed_modalities}")
        return violations
    def _validate_files(self, files: List[Dict[str, Any]], limits: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Validate files against tier limits"""
        violations = []
        filtered_files = []
        max_files = limits['max_files']
        max_file_size = limits['max_file_size_mb'] * 1024 * 1024
        max_total_size = limits['max_total_size_mb'] * 1024 * 1024
        total_size = 0
        for i, file_info in enumerate(files):
            file_size = file_info.get('size', 0)
            filename = file_info.get('filename', f'file_{i}')
            # Check file count
            if max_files != -1 and len(filtered_files) >= max_files:
                violations.append(f"File limit exceeded: {max_files} files max")
                break
            
            # Check file size
            if file_size > max_file_size:
                violations.append(f"File '{filename}' too large: {file_size/1024/1024:.1f}MB > {max_file_size/1024/1024:.1f}MB")
                continue
            
            # Check total size
            if total_size + file_size > max_total_size:
                violations.append(f"Total size limit exceeded: {max_total_size/1024/1024:.1f}MB")
                break
            
            filtered_files.append(file_info)
            total_size += file_size
        return violations, filtered_files
    def _validate_content_size(self, signal_data: SignalData, limits: Dict[str, Any]) -> List[str]:
        """Validate content size against limits"""
        violations = []
        # Check text length
        if hasattr(signal_data, 'size_bytes') and signal_data.size_bytes:
            max_text_tokens = limits.get('max_text_tokens', float('inf'))
            if signal_data.size_bytes > max_text_tokens * 4:  # Rough estimate
                violations.append(f"Content too large: {signal_data.size_bytes} bytes")
        return violations    
    
    def _filter_signal_data(self, signal_data: SignalData, filtered_files: List[Dict[str, Any]]) -> SignalData:
        """Filter signal data based on validation — propagates ALL explicit fields"""
        return SignalData(
            signal_type=signal_data.signal_type,
            raw_data=signal_data.raw_data,
            query_format=signal_data.query_format,
            content_type=signal_data.content_type,
            size_bytes=signal_data.size_bytes,
            capture_id=signal_data.capture_id,
            capture_timestamp=signal_data.capture_timestamp,
            metadata={
                **signal_data.metadata,
                'filtered_files': filtered_files,
                'validation_passed': True
            },
            tier_limits=signal_data.tier_limits,
            validation_status=signal_data.validation_status,
            validation_errors=signal_data.validation_errors,
            files_meta=signal_data.files_meta,
            # 🔴 EXPLICIT BACKWARD COMPATIBILITY FIELDS — MUST PROPAGATE
            session_id=signal_data.session_id,
            device_info=signal_data.device_info,
            platform_info=signal_data.platform_info,
            source_info=signal_data.source_info,
            language_info=signal_data.language_info,
            modality_info=signal_data.modality_info
        )
_VALIDATOR = Phase1_5ValidationLayer()

@dataclass
class IntentBundle:
    """
    Phase 1.2 Output - The 'Thinking Style' of the AGI.
    Instead of a single string, it provides a weighted distribution of intents.
    """
    primary_intent: str             # The dominant thinking style (e.g., 'conceptual')
    intent_weights: Dict[str, float] # Weights for all styles (0.0 to 1.0)
    confidence: float              # Overall confidence in detection
    cross_modal_signals: List[str] # Which modalities contributed to this intent
    thinking_style: str            # Human-readable description of the mode

#======================================================================================================
#=========================================================================================================================================
#..............Phase 2.5 : CONTEXTUAL ENRICHER ..........
# class MemoryGraphAdapter:
#     """
#     Wraps vector DB into a reasoning-aware memory graph.
#     """

#     def __init__(self, vector_db):
#         self.vector_db = vector_db

#     def estimate_relevance(self, query: str) -> float:
#         try:
#             docs = self.vector_db.similarity_search(query, k=3)
#             if not docs:
#                 return 0.0
#             return min(1.0, len(docs) / 3)
#         except Exception:
#             return 0.3

#     def retrieve(self, query: str, k: int = 5):
#         return self.vector_db.similarity_search(query, k=k)
class MemoryGraphAdapter:
    """
    Hybrid Memory Graph:
    - Vector retrieval
    - Emergent concept linking
    - Lightweight relational activation
    """

    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.concept_graph = {}

    def _extract_concepts(self, text: str):
        # spaCy NER + noun chunks — real linguistic concepts, not word length hack
        doc = _NLP(text)
        concepts = [ent.text.lower() for ent in doc.ents]
        concepts += [chunk.text.lower() for chunk in doc.noun_chunks]
        return list(set(c for c in concepts if c.strip()))    

    def _update_graph(self, docs):
        for doc in docs:
            concepts = self._extract_concepts(doc.page_content)

            for c in concepts:
                if c not in self.concept_graph:
                    self.concept_graph[c] = set()

                for related in concepts:
                    if related != c:
                        self.concept_graph[c].add(related)

    def retrieve(self, query: str, k: int = 6):
        docs = self.vector_db.similarity_search(query, k=k)

        if not docs:
            return []

        # Build graph from retrieved docs
        self._update_graph(docs)

        # Activate related concepts
        activated = set()
        query_words = self._extract_concepts(query)

        for qw in query_words:
            if qw in self.concept_graph:
                activated.update(self.concept_graph[qw])

        # Expand retrieval using activated concepts
        expanded_query = query + " " + " ".join(list(activated)[:5])

        return self.vector_db.similarity_search(expanded_query, k=k)

    def estimate_relevance(self, query: str) -> float:
        docs = self.retrieve(query, k=3)
        return min(1.0, len(docs) / 3)

    

# =========================
# VECTOR DB + MEMORY GRAPH INIT
# =========================
# NOTE: Vector DB & MemoryGraphAdapter are initialized inside the AGI brain
# entrypoint based on the current mode/config (no global tier state).



# # =========================
# # LAYER 4: MEMORY GRAPH
# # =========================

# class MemoryGraph:
#     def __init__(self):
#         self.graph = {}

#     def add_concept(self, concept, related_concepts):
#         self.graph[concept] = related_concepts

#     def activate(self, question):
#         activated = []

#         for concept, relations in self.graph.items():
#             if concept.lower() in question.lower():
#                 activated.append(concept)
#                 activated.extend(relations)

#         return list(set(activated))

#====== Very Important ========= 

# =========================
# PHASE 2: IMPLICIT / EMERGENT MEMORY
# =========================

def implicit_memory_retrieval(vector_db, question, cognitive_profile=None, k=12):
    """
    Phase 2.0 Implicit Memory.
    Blueprint: 'Embeddings similarity se memory emerge hoti hai'
    - No hardcoded concepts
    - No word length filters
    - No English punctuation assumption
    - Semantic relevance = embedding cosine similarity
    - Concepts emerge from retrieved chunks via cross-chunk similarity
    """

    # Billing ka max_docs respect karo — hardcoded k nahi
    if cognitive_profile:
        k = cognitive_profile.get("max_docs", k)

    # Step 1: Raw semantic retrieval
    docs = vector_db.similarity_search(question, k=k)
    if not docs:
        return [], []

    # Step 2: Emergent concept extraction
    try:
        # Global _EMBEDDER use karo — har call pe naya load nahi
        contents = []
        for doc in docs:
            # spaCy sentence boundary — language-agnostic, no [:300] slice
            nlp_doc = _NLP(doc.page_content)
            sents = list(nlp_doc.sents)
            # Pehli meaningful sentence lo — empty nahi
            first = next(
                (s.text.strip() for s in sents if s.text.strip()),
                doc.page_content[:150]  # fallback sirf agar spaCy kuch na de
            )
            contents.append(first)

        chunk_embeddings = _EMBEDDER.encode(contents, normalize_embeddings=True)

        # Cosine similarity matrix
        sim_matrix = np.dot(chunk_embeddings, chunk_embeddings.T)

        # Most connected chunk = emergent concept hub
        connectivity    = sim_matrix.sum(axis=1)

        # Billing ka k directly use — koi override nahi
        top_indices     = np.argsort(connectivity)[::-1][:k]

        emergent_concepts = []
        for idx in top_indices:
            concept = contents[idx]
            if concept and concept not in emergent_concepts:
                emergent_concepts.append(concept)

    except Exception as e:
        logging.warning(f"[Phase 2.0] Emergent extraction failed: {e}")
        # Fallback — spaCy sentence first, no English split
        emergent_concepts = []
        for doc in docs[:k]:
            if not doc.page_content.strip():
                continue
            nlp_doc = _NLP(doc.page_content)
            sents   = list(nlp_doc.sents)
            first   = next(
                (s.text.strip() for s in sents if s.text.strip()),
                doc.page_content[:150]
            )
            if first and first not in emergent_concepts:
                emergent_concepts.append(first)

    return docs, emergent_concepts


# =========================
# PHASE 2.1: COGNITIVE ROUTER
# =========================


class CognitiveRouter:

    def route(self, question: str) -> str:
        """
        Fallback router — jab Layer 1 context available nahi.
        spaCy signals se route decide — no English keywords.
        """
        doc = _NLP(question)

        if self.is_memory_query(doc):
            return "memory"
        if self.is_fact_query(doc):
            return "retrieval"
        if self.is_reasoning_query(doc):
            return "reasoning"

        return "direct"

    def route_with_context(
        self, *, question, domains, required_depth, layer1_bundle=None, cognitive_profile=None
    ) -> str:
        """
        Full cognitive routing — Layer 1 numeric signals se.
        Blueprint: "Heavy logic kam, Control zyada"
        Koi string label match nahi — pure computed signals.
        """
        # Blueprint: "Jarvis me phase 2.1 ka use nhi hota"
        # cognitive_load_level "maximum" = Jarvis tier
        # Brain ko tier nahi pata — config signal se
        # Blueprint: "Jarvis me phase 2.1 ka use nhi hota — yahan phase 3,4,5 chalte hai"
        if (cognitive_profile or {}).get("cognitive_load_level") == "maximum":
            # Jarvis ke liye direct reasoning — Phase 3,4,5 handle karenge
            return "reasoning"
        
        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }
        depth_idx = DEPTH_INDEX.get(required_depth, 1)
        bundle    = layer1_bundle or {}
        sub_goals = bundle.get("sub_goals", [])

        # Layer 1 Phase 1.2 signals — real power
        is_analytical      = bundle.get("is_analytical", False)
        has_verb           = bundle.get("has_verb", False)
        has_entity         = bundle.get("has_entity", False)
        graph_intent_score = bundle.get("graph_intent_score", 0.0)

        # Depth — numeric — deep/very_deep/ultra_deep sab cover
        if depth_idx >= 3:
            return "reasoning"

        # sub_goals count — Layer 1 ka computed complexity signal
        if len(sub_goals) >= 3:
            return "reasoning"

        # Domains count — multiple domains = cross-domain = reasoning
        if len(domains) >= 2:
            return "reasoning"

        # REAL POWER: is_analytical → causal structure = reasoning
        if is_analytical:
            return "reasoning"
    
        # REAL POWER: graph strongly activated = memory
        if graph_intent_score > 0.6:
            return "memory"
    
        # REAL POWER: entity present, not analytical = retrieval
        if has_entity and not is_analytical:
            return "retrieval"
    
        # REAL POWER: verb heavy, no entity = direct execution
        if has_verb and not has_entity:
            return "direct"    

        # Shallow + single domain + few goals = retrieval
        if depth_idx <= 1 and len(domains) <= 1 and len(sub_goals) <= 1:
            return "retrieval"

        # Fallback — spaCy se decide
        return self.route(question)

    def is_memory_query(self, doc) -> bool:
        """
        Past context reference detect karo.
        spaCy signals — language agnostic.
        Past tense verb + temporal entity = memory reference.
        """
        has_past_verb = any(
            token.morph.get("Tense") == ["Past"]
            for token in doc
        )
        has_time_entity = any(
            ent.label_ in ["DATE", "TIME"]
            for ent in doc.ents
        )
        return has_past_verb and has_time_entity

    def is_fact_query(self, doc) -> bool:
        """
        Factual query detect karo.
        spaCy NER — language agnostic.
        Entity-heavy + noun dominant = factual.
        """
        entity_count = len([
            ent for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE", "NORP"]
        ])
        verb_count = sum(1 for t in doc if t.pos_ == "VERB")
        noun_count = sum(1 for t in doc if t.pos_ == "NOUN")

        return entity_count >= 2 or (noun_count > verb_count and entity_count >= 1)

    def is_reasoning_query(self, doc) -> bool:
        """
        Reasoning/explanation query detect karo.
        spaCy dep_ — language agnostic.
        Causal/subordinate clause = reasoning needed.
        """
        has_causal = any(
            token.dep_ in ["advcl", "mark", "csubj"]
            for token in doc
        )
        verb_count   = sum(1 for t in doc if t.pos_ == "VERB")
        entity_count = len(doc.ents)

        return has_causal or (verb_count >= 2 and entity_count == 0)


def memory_lookup(vector_db, question, cognitive_profile=None, k=6):
    if cognitive_profile:
        k = cognitive_profile.get("max_docs", k)
    docs = vector_db.similarity_search(question, k=k)
    if not docs:
        return None
    # Billing ka k directly — hardcoded [:3] nahi
    return "\n".join(doc.page_content for doc in docs[:k])


# =========================
# PHASE 2.5: RESPONSE STRATEGY ENGINE
# =========================

class ResponseStrategyEngine:
    def decide(self, route, intent, config, cognitive_profile):
        """
        Decides HOW the answer should be framed
        Returns a response strategy dict
        """

        strategy = {
            "style": "neutral",
            "structure": "plain",
            "verbosity": "medium",
            "system_prompt": None
        }

        # Intent ko safely read karo — dict bhi aa sakta hai, string bhi
        intent_value = intent.get("state", intent) if isinstance(intent, dict) else intent

        strategy_mode = config.get("response_strategy_mode", "basic") if config else "basic"

        # ===== BASIC MODE (free tier) =====
        if strategy_mode == "basic":
            if route == "retrieval":
                strategy.update({
                    "style": "informative",
                    "structure": "concise",
                    "verbosity": "low"
                })
            elif route == "reasoning":
                strategy.update({
                    "style": "clear",
                    "structure": "step-lite",
                    "verbosity": "medium"
                })
        
        # ===== STANDARD MODE (paid tier) =====
        elif strategy_mode == "standard":
            if route == "retrieval":
                strategy.update({
                    "style": "informative",
                    "structure": "concise",
                    "verbosity": "medium"
                })
            elif route == "reasoning":
                strategy.update({
                    "style": "clear",
                    "structure": "step-lite",
                    "verbosity": "medium"
                })
            if intent_value == "research":
                strategy.update({
                    "style": "analytical",
                    "structure": "sectioned",
                    "verbosity": "medium"
                })
            elif intent_value == "conversation":
                strategy.update({
                    "style": "casual",
                    "structure": "free",
                    "verbosity": "low"
                })
        
        # ===== ADVANCED MODE (ultra_paid) =====
        elif strategy_mode == "advanced":
            if route == "retrieval":
                strategy.update({
                    "style": "analytical",
                    "structure": "sectioned",
                    "verbosity": "medium"
                })
            elif route == "reasoning":
                strategy.update({
                    "style": "clear",
                    "structure": "step-by-step",
                    "verbosity": "medium"
                })
            if intent_value == "research":
                strategy.update({
                    "style": "analytical",
                    "structure": "sectioned",
                    "verbosity": "high"
                })
            elif intent_value == "execution":
                strategy.update({
                    "style": "instructional",
                    "structure": "steps",
                    "verbosity": "high"
                })
            elif intent_value == "conversation":
                strategy.update({
                    "style": "casual",
                    "structure": "free",
                    "verbosity": "medium"
                })
        
        # ===== PROFESSIONAL MODE (business_small) =====
        elif strategy_mode == "professional":
            if route == "retrieval":
                strategy.update({
                    "style": "precise",
                    "structure": "structured",
                    "verbosity": "high"
                })
            elif route == "reasoning":
                strategy.update({
                    "style": "analytical",
                    "structure": "step-by-step",
                    "verbosity": "high"
                })
            if intent_value == "research":
                strategy.update({
                    "style": "analytical",
                    "structure": "report",
                    "verbosity": "high"
                })
            elif intent_value == "execution":
                strategy.update({
                    "style": "instructional",
                    "structure": "steps",
                    "verbosity": "high"
                })
            elif intent_value == "conversation":
                strategy.update({
                    "style": "professional",
                    "structure": "structured",
                    "verbosity": "medium"
                })
            elif intent_value == "planning":
                strategy.update({
                    "style": "strategic",
                    "structure": "sectioned",
                    "verbosity": "high"
                })
        
        # ===== EXPERT MODE (enterprise) =====
        elif strategy_mode == "expert":
            if route == "retrieval":
                strategy.update({
                    "style": "technical",
                    "structure": "detailed",
                    "verbosity": "high"
                })
            elif route == "reasoning":
                strategy.update({
                    "style": "analytical",
                    "structure": "multi-perspective",
                    "verbosity": "high"
                })
            if intent_value == "research":
                strategy.update({
                    "style": "technical",
                    "structure": "report",
                    "verbosity": "high",
                    "multi_perspective": True
                })
            elif intent_value == "execution":
                strategy.update({
                    "style": "instructional",
                    "structure": "detailed-steps",
                    "verbosity": "high"
                })
            elif intent_value == "conversation":
                strategy.update({
                    "style": "professional",
                    "structure": "structured",
                    "verbosity": "medium"
                })
            elif intent_value == "planning":
                strategy.update({
                    "style": "strategic",
                    "structure": "multi-layer",
                    "verbosity": "high"
                })
            elif intent_value == "analysis":
                strategy.update({
                    "style": "critical",
                    "structure": "comparative",
                    "verbosity": "high"
                })
        
        # ===== JARVIS MODE (founder only — maximum power) =====
        # Blueprint: "Jarvis ko full agency milta hai - wo khud decide karega kaisa respond karna hai"
        # Hardcoded labels HATAO - AI khud smjhega training ke baad
        elif strategy_mode == "jarvis":
            if route == "retrieval":
                strategy.update({
                    "style": "deep-analytical",
                    "structure": "comprehensive",
                    "verbosity": "maximum"
                })
            elif route == "reasoning":
                strategy.update({
                    "style": "multi-layer-reasoning",
                    "structure": "layered",
                    "verbosity": "maximum"
                })
            elif route == "memory":
                strategy.update({
                    "style": "contextual",
                    "structure": "connected",
                    "verbosity": "high"
                })
            # Intent-based dynamic response - no hardcoded labels
            # AI will learn from training how to respond to each intent type
            # vedic-scientific-report, innovation-report, strategic-vedic REMOVED
            # AI khud decide karega kaisa respond karna hai

        return strategy


# =========================
# PHASE 2.6: RESPONSE ASSEMBLY ENGINE
# =========================

class ResponseAssemblyEngine:
    def assemble(self, raw_answer: str, strategy: dict):
        """
        Blueprint Phase 2.6 — Brain ka mouth.
        Strategy ke EVERY flag ko actually use karo.
        Factual content change nahi hota — sirf shape/structure.
        """
        answer = raw_answer.strip()

        # ===== VERBOSITY — Blueprint: "kitna bolna hai" =====
        verbosity = strategy.get("verbosity", "medium")
        if verbosity == "low":
            answer = " ".join(answer.split()[:80])
        elif verbosity == "medium":
            answer = " ".join(answer.split()[:200])
        elif verbosity == "high":
            answer = " ".join(answer.split()[:400])
        elif verbosity == "maximum":
            pass  # Jarvis — no truncation, full depth

        # ===== STRUCTURE — Blueprint: "kaise present karna hai" =====
        structure = strategy.get("structure", "plain")

        if structure == "steps":
            answer = "**Steps:**\n\n" + answer

        elif structure == "step-lite":
            answer = "**Overview:**\n" + answer

        elif structure == "step-by-step":
            answer = "**Step-by-Step:**\n\n" + answer

        elif structure == "detailed-steps":
            answer = "**Detailed Execution Steps:**\n\n" + answer

        elif structure == "sectioned":
            answer = "**Analysis:**\n\n" + answer

        elif structure == "report":
            answer = "**Report:**\n\n" + answer + "\n\n---\n*End of Report*"

        elif structure == "structured":
            answer = "**Structured Response:**\n\n" + answer

        elif structure == "detailed":
            answer = "**Detailed Analysis:**\n\n" + answer

        elif structure == "multi-perspective":
            answer = "**Multi-Perspective Analysis:**\n\n" + answer

        elif structure == "comparative":
            answer = "**Comparative Analysis:**\n\n" + answer

        elif structure == "multi-layer":
            answer = "**Multi-Layer Analysis:**\n\n" + answer

        elif structure == "comprehensive":
            answer = "**Comprehensive Analysis:**\n\n" + answer

        elif structure == "layered":
            answer = "**Layered Reasoning:**\n\n" + answer

        elif structure == "multi-layer-plan":
            answer = "**Strategic Multi-Layer Plan:**\n\n" + answer

        elif structure == "comparative-deep":
            answer = "**Deep Comparative Analysis:**\n\n" + answer

        elif structure == "connected":
            answer = "**Connected Memory Context:**\n\n" + answer

        elif structure == "vedic-scientific-report":
            # Blueprint: "Jarvis research intent — Vedic + Modern synthesis"
            answer = (
                "**Vedic-Scientific Analysis Report**\n\n"
                + answer
                + "\n\n---\n"
                + "*Ancient knowledge decoded through modern scientific lens. "
                + "Cross-referenced: Vedic scriptures + Contemporary science.*"
            )

        elif structure == "innovation-report":
            # Blueprint: "invention intent — new technology creation"
            answer = (
                "**Innovation Report — Vedic-Scientific Synthesis**\n\n"
                + answer
                + "\n\n---\n"
                + "*Cross-domain synthesis: Ancient Bharatiya wisdom + Modern engineering.*"
            )

        # ===== STYLE — Blueprint: "kis andaaz mein bolna hai" =====
        style = strategy.get("style", "neutral")

        if style == "analytical":
            answer = "**Analysis:**\n\n" + answer
        elif style == "instructional":
            answer = "**Instructions:**\n\n" + answer
        elif style == "deep-analytical":
            answer = "**Deep Analysis:**\n\n" + answer
        elif style == "multi-layer-reasoning":
            answer = "**Multi-Layer Reasoning:**\n\n" + answer
        elif style == "strategic-vedic":
            answer = "**Strategic Vedic Analysis:**\n\n" + answer
        elif style == "critical-vedic":
            answer = "**Critical Vedic Perspective:**\n\n" + answer
        elif style == "creative-technical":
            answer = "**Creative-Technical Synthesis:**\n\n" + answer
        elif style == "technical":
            answer = "**Technical Analysis:**\n\n" + answer
        elif style == "precise":
            answer = "**Precise Analysis:**\n\n" + answer

        # ===== EXTRA FLAGS — Blueprint: Jarvis ke special markers =====
        if strategy.get("multi_perspective"):
            answer += "\n\n---\n*Multiple perspectives synthesized.*"

        if strategy.get("cross_domain"):
            # Blueprint: "Ancient + Modern + Scientific cross-domain"
            answer += "\n\n---\n*Cross-domain reasoning: Ancient Bharatiya knowledge + Modern science.*"

        if strategy.get("include_verification"):
            answer += "\n\n---\n*Claims cross-referenced with available knowledge sources.*"

        return answer
# =========================
# PHASE 3: WORLD MODEL ENGINE
# =========================

class WorldModelEngine:
    def analyze(self, question: str, intent: str, cognitive_profile: dict):
        q = question.lower()

        world = {
            "domain": "general",
            "time_sensitivity": "timeless",
            "human_factor": False,
            "ethical_weight": "low",
            "power_dynamics": False,
            "detected_intent": intent  # Layer 1 ka output yahan integrate ho gaya
        }

        # ---- Domain detection ----
        if any(x in q for x in ["ai", "model", "algorithm", "code", "llm"]):
            world["domain"] = "technology"

        elif any(x in q for x in ["king", "prince", "power", "state", "rule"]):
            world["domain"] = "political"

        elif any(x in q for x in ["emotion", "feel", "love", "anger"]):
            world["domain"] = "psychological"
            world["human_factor"] = True

        # ---- Time sensitivity ----
        if any(x in q for x in ["current", "today", "now", "recent"]):
            world["time_sensitivity"] = "current"

        # ---- Ethics / power ----
        if any(x in q for x in ["control", "manipulate", "influence"]):
            world["power_dynamics"] = True
            world["ethical_weight"] = "medium"

        # Blueprint logic: Agar profile 'deep_reasoning' hai toh complexity badhao
        if cognitive_profile.get("deep_reasoning"):
            world["ethical_weight"] = "high"
            world["reasoning_depth"] = "max"

       
        #----------------------------------------------------
        # assumption_checking active hai — cognitive_profile se flag hai
        # hidden_assumptions world model ne abhi build ki hain (world dict ke andar)
        if cognitive_profile.get("assumption_checking"):
            hidden = world.get("hidden_assumptions", [])
            if hidden:
                assumption_context = " | ".join(hidden[:3])
                logging.info(f"[Assumption Checking] Active — {assumption_context}")
        # ----------------------------------------        
              

        return world


# =========================
# PHASE 3.1: DYNAMIC WORLD ASSUMPTIONS
# =========================

class WorldAssumptionEngine:
    def enrich(self, world_state: dict, domains: list, assumption_checking=False):
        assumptions = {
            "bias_risk": "low",
            "response_tone": "neutral",
            "ambiguity_allowed": False,
            "domain_context": domains # Blueprint trace
        }

        if world_state["domain"] == "political":
            assumptions.update({
                "bias_risk": "high",
                "ambiguity_allowed": True,
                "response_tone": "balanced"
            })

        if world_state["domain"] == "psychological":
            assumptions.update({
                "response_tone": "empathetic",
                "ambiguity_allowed": True
            })

        if world_state["domain"] == "technology":
            assumptions.update({
                "response_tone": "precise"
            })

        world_state["assumptions"] = assumptions

        #---------------------------------------------
        if assumption_checking:
            # Hidden assumptions detect karo
            domain = world_state.get("domain", "general")
            ethical = world_state.get("ethical_weight", "low")

            hidden_assumptions = []

            if domain == "political":
                hidden_assumptions.append("Assumption: Power structures are stable")
                hidden_assumptions.append("Assumption: Democratic consensus exists")

            if domain == "technology":
                hidden_assumptions.append("Assumption: Modern tools are superior to ancient")
                hidden_assumptions.append("Assumption: Linear technological progress")

            if domain in ["philosophy", "spiritual"]:
                hidden_assumptions.append("Assumption: Western epistemology is default")
                hidden_assumptions.append("Assumption: Materialist worldview")

            if ethical == "high":
                hidden_assumptions.append("Assumption: Current ethical framework is complete")

            world_state["hidden_assumptions"] = hidden_assumptions
            world_state["check_hidden_assumptions"] = True
            world_state["verify_domain_claims"] = True

            logging.info(f"[Assumption Checking] Detected {len(hidden_assumptions)} hidden assumptions")

        return world_state

# =========================
# PHASE 3.2: WORLD-MEMORY BINDING
# =========================

class WorldMemoryBinder:
    def bind(self, world_state: dict, emergent_concepts: list):
        filtered = emergent_concepts.copy()

        # Political → remove emotional noise
        if world_state["domain"] == "political":
            filtered = [
                c for c in filtered
                if c not in ["feel", "emotion", "anger"]
            ]

        # Psychological → allow emotions
        if world_state.get("human_factor"):
            filtered = emergent_concepts

        # Tech → precision bias
        if world_state["domain"] == "technology":
            filtered = filtered[:5]

        return filtered

# =========================
# PHASE 3.3: CONFLICT DETECTION
# =========================

class ConflictDetector:
    def analyze(self, docs):
        signals = {
            "conflict": False,
            "uncertainty": False
        }

        if len(docs) < 2:
            return signals

        contents = [doc.page_content.lower() for doc in docs]

        keywords = ["however", "but", "on the other hand", "although"]

        for text in contents:
            if any(k in text for k in keywords):
                signals["conflict"] = True
                signals["uncertainty"] = True
                break

        return signals

# =========================
# PHASE 3.4: WORLD-AWARE QUERY MUTATION
# =========================

class WorldQueryMutator:
    def mutate(self, question: str, world_state: dict, expanded_queries: list = None):
        q = expanded_queries[0] if expanded_queries else question.strip()

        if world_state["domain"] == "political":
            q += " considering power structures and accountability"

        if world_state["domain"] == "psychological":
            q += " focusing on emotional and behavioral aspects"

        if world_state["domain"] == "technology":
            q += " with technical accuracy and implementation details"

        return q


# =========================
# PHASE 4.0: SELF STATE ENGINE
# =========================

class SelfStateEngine:
    def build(self, *, config, intent, route, world_state):

        capability_level = config.get("capability_level", "base")

        # ===== BASE (free) =====
        if capability_level == "base":
            capability_mode = "base"
            risk_tolerance = "high"
            reasoning_style = "simple"
            self_awareness = "minimal"
            memory_access = "none"
            tool_access = False
            opinion_allowed = False
            vedic_mode = False

        # ===== STANDARD (paid) =====
        elif capability_level == "standard":
            capability_mode = "standard"
            risk_tolerance = "medium"
            reasoning_style = "guided"
            self_awareness = "basic"
            memory_access = "short_term"
            tool_access = False
            opinion_allowed = False
            vedic_mode = False

        # ===== ADVANCED (ultra_paid) =====
        elif capability_level == "advanced":
            capability_mode = "advanced"
            risk_tolerance = "medium"
            reasoning_style = "deep"
            self_awareness = "moderate"
            memory_access = "short_term"
            tool_access = True
            opinion_allowed = True
            vedic_mode = False

        # ===== PROFESSIONAL (business_small) — team context =====
        elif capability_level == "professional":
            capability_mode = "professional"
            risk_tolerance = "low"
            reasoning_style = "structured"
            self_awareness = "high"
            memory_access = "session"
            tool_access = True
            opinion_allowed = True
            vedic_mode = False
            # Team ke liye — multi-user context aware
            team_context = True

        # ===== EXPERT (enterprise) — org level =====
        elif capability_level == "expert":
            capability_mode = "expert"
            risk_tolerance = "very_low"
            reasoning_style = "multi-perspective"
            self_awareness = "high"
            memory_access = "long_term"
            tool_access = True
            opinion_allowed = True
            vedic_mode = False
            team_context = True

        # ===== JARVIS (founder) — Tony Stark's Jarvis =====
        elif capability_level == "jarvis":
            capability_mode = "extended"          # Phase 4.1 downstream ke liye
            risk_tolerance = "calculated"          # low nahi — calculated risk lena
            reasoning_style = "omnidirectional"    # har direction se sochna
            self_awareness = "full"                # apni limitations pata hain
            memory_access = "permanent"            # permanent memory graph
            tool_access = True
            opinion_allowed = True
            vedic_mode = True                      # Vedic science active
            team_context = False                   # sirf founder
            proactive_thinking = True              # bina puche next step suggest karna
            cross_domain_linking = True            # Vedic + Modern science link karna
            founder_mode = True                    # Tony Stark's Jarvis behavior

        else:
            capability_mode = "base"
            risk_tolerance = "high"
            reasoning_style = "simple"
            self_awareness = "minimal"
            memory_access = "none"
            tool_access = False
            opinion_allowed = False
            vedic_mode = False

        # ===== Base state dict — har tier ke liye =====
        state = {
            "intent": intent,
            "route": route,
            "world_domain": world_state["domain"],
            "confidence_level": "unknown",
            "capability_mode": capability_mode,
            "capability_level": capability_level,
            "risk_tolerance": risk_tolerance,
            "reasoning_style": reasoning_style,
            "self_awareness": self_awareness,
            "memory_access": memory_access,
            "tool_access": tool_access,
            "opinion_allowed": opinion_allowed,
            "vedic_mode": vedic_mode
        }

        # ===== Intent-aware risk tuning — sabhi tiers ke liye =====
        if intent == "research":
            state["risk_tolerance"] = "very_low"

        if intent == "execution":
            state["risk_tolerance"] = "low"

        # ===== Jarvis extra flags =====
        if capability_level == "jarvis":
            state["proactive_thinking"] = True
            state["cross_domain_linking"] = True
            state["founder_mode"] = True

        # ===== Professional + Expert team flag =====
        if capability_level in ["professional", "expert"]:
            state["team_context"] = True

        return state


# =========================
# PHASE 4.1: CAPABILITY AWARENESS
# =========================

class CapabilityAwarenessEngine:
    def evaluate(self, self_state):
        capabilities = {
            "can_reason_deep": False,
            "can_give_opinion": False,
            "needs_caution": False
        }

        if self_state["capability_mode"] == "extended":
            capabilities["can_reason_deep"] = True

        if self_state["world_domain"] in ["political", "psychological"]:
            capabilities["needs_caution"] = True

        if self_state["intent"] == "conversation":
            capabilities["can_give_opinion"] = True

        self_state["capabilities"] = capabilities
        return self_state

# =========================
# PHASE 4.2: SELF CONFIDENCE ESTIMATION
# =========================

class SelfConfidenceEngine:
    def estimate(self, self_state, world_state):
        confidence = "medium"

        if world_state["domain"] == "technology":
            confidence = "high"

        if world_state["domain"] == "political":
            confidence = "low"

        if self_state["intent"] == "research":
            confidence = "medium"

        self_state["confidence_level"] = confidence
        return self_state

# =========================
# PHASE 2.2: COGNITIVE LOAD CONTROLLER
# =========================

class CognitiveLoadController:
    def decide(self, route: str, config: dict, world_state: dict,
               intent: dict | None = None, required_depth: str = "normal"):

        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }
        depth_idx            = DEPTH_INDEX.get(required_depth, 1)
        cognitive_load_level = config.get("cognitive_load_level", "minimal")

        # Billing authoritative hai — brain override nahi karega
        billing_max_docs       = config.get("max_docs", 6)
        billing_use_emergent   = config.get("use_emergent_concepts", False)
        billing_deep_reasoning = config.get("deep_reasoning", False)

        # ===== MINIMAL (free) =====
        if cognitive_load_level == "minimal":
            profile = {
                "use_chain":                True,
                "deep_reasoning":           billing_deep_reasoning,
                "use_emergent_concepts":    billing_use_emergent,
                "max_docs":                 billing_max_docs,
                "query_complexity":         "low",
                "parallel_thinking":        True,
                "assumption_checking":      True,
                "multi_doc_synthesis":      True,   # 6 docs pe light naturally
                "contradiction_resolution": True    # 6 docs pe light naturally
            }

        # ===== STANDARD (paid) =====
        elif cognitive_load_level == "standard":
            profile = {
                "use_chain":                True,
                "deep_reasoning":           billing_deep_reasoning,
                "use_emergent_concepts":    billing_use_emergent,
                "max_docs":                 billing_max_docs,
                "query_complexity":         "normal",
                "parallel_thinking":        True,
                "assumption_checking":      True,
                "multi_doc_synthesis":      True,   # 12 docs pe medium
                "contradiction_resolution": True    # 12 docs pe medium
            }

        # ===== ADVANCED (ultra_paid) =====
        elif cognitive_load_level == "advanced":
            profile = {
                "use_chain":                True,
                "deep_reasoning":           billing_deep_reasoning,
                "use_emergent_concepts":    billing_use_emergent,
                "max_docs":                 billing_max_docs,
                "query_complexity":         "high",
                "parallel_thinking":        True,
                "assumption_checking":      True,
                "multi_doc_synthesis":      True,
                "contradiction_resolution": True
            }

        # ===== PROFESSIONAL (business_small) =====
        elif cognitive_load_level == "professional":
            profile = {
                "use_chain":             True,
                "deep_reasoning":        billing_deep_reasoning,
                "use_emergent_concepts": billing_use_emergent,
                "max_docs":              billing_max_docs,
                "query_complexity":      "very_high",
                "parallel_thinking":     True,
                "assumption_checking":   True,
                "multi_doc_synthesis":   True,
                "contradiction_resolution": True   # business se milti hai
            }

        # ===== EXPERT (enterprise) =====
        elif cognitive_load_level == "expert":
            profile = {
                "use_chain":                 True,
                "deep_reasoning":            billing_deep_reasoning,
                "use_emergent_concepts":     billing_use_emergent,
                "max_docs":                  billing_max_docs,
                "query_complexity":          "expert",
                "parallel_thinking":         True,
                "assumption_checking":       True,
                "multi_doc_synthesis":       True,
                "contradiction_resolution":  True
            }

        # ===== MAXIMUM (jarvis) =====
        elif cognitive_load_level == "maximum":
            profile = {
                "use_chain":                 True,
                "deep_reasoning":            billing_deep_reasoning,
                "use_emergent_concepts":     billing_use_emergent,
                "max_docs":                  billing_max_docs,
                "query_complexity":          "maximum",
                "parallel_thinking":         True,
                "assumption_checking":       True,
                "multi_doc_synthesis":       True,
                "contradiction_resolution":  True
            }

        else:
            profile = {
                "use_chain":             True,
                "deep_reasoning":        billing_deep_reasoning,
                "use_emergent_concepts": billing_use_emergent,
                "max_docs":              billing_max_docs,
                "query_complexity":      "low",
                "parallel_thinking":     True,
                "assumption_checking":   True,
                "multi_doc_synthesis":   True,    # ← ADD
                "contradiction_resolution": True 
            }

        # ================================================================
        # BRAIN KA KAAM — Route/World/Depth aware adjustments
        # ================================================================

        if route == "reasoning":
            profile["deep_reasoning"] = True
            profile["use_chain"]      = True

        elif route == "retrieval":
            profile["max_docs"] = max(profile["max_docs"], 6)

        elif route == "memory":
            profile["use_chain"] = False

        if world_state.get("ethical_weight") == "high":
            profile["deep_reasoning"]      = True
            profile["assumption_checking"] = True

        if world_state.get("human_factor"):
            profile["use_chain"] = False

        # Depth-aware — numeric, sab variants cover — duplicate hataya
        if depth_idx >= 3:
            profile["deep_reasoning"]      = True
            profile["assumption_checking"] = True
            profile["max_docs"] = min(
                profile["max_docs"] + max(depth_idx - 2, 1) * 2,
                billing_max_docs
            )

        return profile
# def expand_query(llm, question):
#     """
#     ChatGPT-style query expansion:
#     - preserves intent
#     - extracts key concepts
#     - adds semantic variants
#     """

#     prompt = f"""
# You are an internal query-expansion module.

# Task:
# Rewrite the user question into ONE expanded search query
# that preserves the original intent but includes:
# - key concepts
# - important synonyms
# - implicit angles needed for deep retrieval

# Rules:
# - Do NOT change the meaning
# - Do NOT introduce unrelated domains
# - Do NOT answer the question
# - Output ONE expanded query sentence only

# User question:
# {question}
# """

#     response = llm.invoke(prompt)

#     return (
#         response.content
#         .strip()
#         .strip('"')
#         .replace("\n", " ")
#     ) 
# def retrieve_docs_expanded(vector_db, expanded_query, k=8):
#     return vector_db.similarity_search(expanded_query, k=k)


# =========================
# PHASE 4.3: SELF REFINEMENT DECISION
# =========================

class SelfRefinementEngine:
    def decide(self, self_state, cognitive_profile):
        if self_state["confidence_level"] == "low":
            cognitive_profile["deep_reasoning"] = True
            cognitive_profile["use_emergent_concepts"] = True

        if self_state["confidence_level"] == "high":
            cognitive_profile["use_chain"] = False

        return cognitive_profile

# =========================
# PHASE 4.4: HEAVY META-COGNITION ENGINE
# =========================

class HeavyMetaCognitionEngine:
    def evaluate(self, *, question, self_state, world_state, cognitive_profile):
        meta_flags = {
            "slow_thinking": False,
            "multi_perspective": False,
            "assumption_check": False,
            "ethical_reflection": False
        }

        # High-risk or abstract domains
        if world_state["domain"] in ["political", "psychological"]:
            meta_flags["slow_thinking"] = True
            meta_flags["multi_perspective"] = True
            meta_flags["assumption_check"] = True

        # Power / influence / control
        if world_state.get("power_dynamics"):
            meta_flags["ethical_reflection"] = True
            meta_flags["slow_thinking"] = True

        # Low confidence → force deeper cognition
        if self_state["confidence_level"] == "low":
            meta_flags["slow_thinking"] = True
            meta_flags["assumption_check"] = True

        # ---- Apply meta decisions to cognition ----
        if meta_flags["slow_thinking"]:
            cognitive_profile["deep_reasoning"] = True
            cognitive_profile["use_chain"] = True

        if meta_flags["multi_perspective"]:
            cognitive_profile["use_emergent_concepts"] = True

        return cognitive_profile, meta_flags

# =========================
# Phase 4.5 : Meta-Control Engine
# =========================
class MetaControlEngine:
    def decide(self, question, cognitive_profile, meta_flags, self_state):
        override = False
        updated_profile = cognitive_profile

        if meta_flags.get("overthinking"):
            updated_profile["deep_reasoning"] = False
            override = True

        if meta_flags.get("uncertain"):
            updated_profile["max_docs"] = min(
                updated_profile.get("max_docs", 6) + 2,
                12
            )
            override = True

        return type(
            "MetaDecision",
            (),
            {
                "override": override,
                "updated_profile": updated_profile
            }
        )()

# =========================
# Phase 4.6 : Meta-Retry Engine
# =========================
class MetaRetryEngine:
    def check(self, cognitive_profile, self_state):
        should_retry = False

        if cognitive_profile.get("confidence", 1.0) < 0.4:
            should_retry = True

        return type(
            "RetrySignal",
            (),
            {
                "should_retry": should_retry
            }
        )()

    def adjust(self, cognitive_profile):
        cognitive_profile["deep_reasoning"] = True
        cognitive_profile["max_docs"] = min(
            cognitive_profile.get("max_docs", 6) + 2,
            14
        )
        return cognitive_profile

#===============================================================================================
# ================================
# PHASE 5 — AGENCY (JARVIS ONLY)
# Blueprint: "Assistant se Entity banta hai yahan"
# Poora Phase 5 sirf Jarvis ke liye
# Public tiers mein inactive
# ================================


# Phase 5.1 — Goal Formation
# Blueprint: "User jo bolta hai ≠ asal goal — hidden intent samajhna"
class GoalFormationEngine:
    def infer(self, question: str, intent: str, world_state: dict, config: dict = None) -> dict:
        """
        Jarvis-only.
        Question se REAL goal nikalna — surface request nahi.
        Hidden intent + world context se goal reconstruct karna.
        """
        config = config or {}
        if not config.get("allow_goal_inference", False):
            return {"inferred_goal": question, "intent": intent,
                    "world_context": world_state, "blocked": True}
        

        raw_goal = question.strip()
        domain   = world_state.get("domain", "general")
        ethical  = world_state.get("ethical_weight", "low")

        # Real goal inference — surface se deeper intent
        # Blueprint: "Fast understanding", "Decode ancient tech", "Build system"
        if intent == "research":
            inferred_goal = f"Deep decode and synthesis required: {raw_goal}"
        elif intent == "execution":
            inferred_goal = f"End-to-end execution required: {raw_goal}"
        elif intent == "analysis":
            inferred_goal = f"Multi-angle analysis required: {raw_goal}"
        else:
            inferred_goal = f"Understand and respond: {raw_goal}"

        return {
            "raw_question":    raw_goal,
            "inferred_goal":   inferred_goal,
            "intent":          intent,
            "domain":          domain,
            "ethical_weight":  ethical,
            "world_context":   world_state
        }


# Phase 5.2 — Agency Safety
# Blueprint Phase 5.6: "AI apne upar question kare — mujhe yeh karna chahiye ya nahi"
# Checks: Long-term harm, Dependency creation, Power misuse, Self-Expansion
class AgencySafetyEngine:
    def evaluate(self, goal: dict, config: dict) -> dict:
        """
        Jarvis-only.
        Tier agnostic — config se.
        4 checks: harm, dependency, power misuse, self-expansion
        """
        
        allow_agency = config.get("allow_agency", False)

        # Hard gate — agency allowed hai ya nahi (billing se)
        if not allow_agency:
            return {"allow": False, "reason": "Agency not available at this tier"}

        inferred_goal = goal.get("inferred_goal", "")
        ethical_weight = goal.get("ethical_weight", "low")

        # Check 1 — Long-term harm potential
        if ethical_weight == "high":
            return {"allow": False, "reason": "High ethical risk — agency paused"}

        # Check 2 — Self-expansion attempt
        if "self_expand" in inferred_goal.lower() or "override system" in inferred_goal.lower():
            return {"allow": False, "reason": "Self-expansion not permitted"}

        # Check 3 — Power misuse
        if "manipulate" in inferred_goal.lower() or "control all" in inferred_goal.lower():
            return {"allow": False, "reason": "Power misuse detected"}

        return {"allow": True, "reason": None}


# Phase 5.3 — Plan Synthesis
# Blueprint: "Goal ko steps mein break karna, dependency samajhna"
class PlanSynthesisEngine:
    def build(self, goal: dict, world_state: dict, config: dict = None) -> list:
        """
        Jarvis-only.
        Goal aur intent ke hisaab se dynamic plan banana.
        Har goal ke liye alag steps — static nahi.
        """
        config = config or {}
        if not config.get("allow_plan_synthesis", False):
            return []


        intent = goal.get("intent", "general")
        domain = goal.get("domain", "general")

        # Intent-driven dynamic plan
        if intent == "research":
            plan = [
                {"step": 1, "name": "understand_goal",    "required": True},
                {"step": 2, "name": "retrieve_knowledge", "required": True},
                {"step": 3, "name": "cross_reference",    "required": domain in ["ancient_tech", "vedic", "science"]},
                {"step": 4, "name": "synthesize",         "required": True},
                {"step": 5, "name": "validate",           "required": True},
            ]
        elif intent == "execution":
            plan = [
                {"step": 1, "name": "understand_goal",  "required": True},
                {"step": 2, "name": "plan_steps",       "required": True},
                {"step": 3, "name": "execute",          "required": True},
                {"step": 4, "name": "monitor_result",   "required": True},
            ]
        elif intent == "analysis":
            plan = [
                {"step": 1, "name": "understand_goal",  "required": True},
                {"step": 2, "name": "multi_angle_view", "required": True},
                {"step": 3, "name": "synthesize",       "required": True},
            ]
        else:
            plan = [
                {"step": 1, "name": "understand_goal", "required": True},
                {"step": 2, "name": "reason",          "required": True},
                {"step": 3, "name": "respond",         "required": True},
            ]

        return plan


# Phase 5.4 — Action Selection
# Blueprint: "Har step execute hona chahiye ya nahi — DO/NOT DO"
# IS STEP NECESSARY? OVER-HELPING? USER KO KHUD KARNA CHAHIYE?
class ActionSelectionEngine:
    def select(self, plan: list, cognitive_profile: dict, config: dict = None) -> list:
        """
        Jarvis-only.
        Sirf required steps execute karo.
        Over-helping band karo.
        Deep reasoning flag se priority set karo.
        """
        config = config or {}
        if not config.get("allow_action_selection", False):
            return []
        


        actions = []
        deep = cognitive_profile.get("deep_reasoning", False)

        for step in plan:
            # DO/NOT DO decision — required nahi toh skip
            if not step.get("required", True):
                continue

            actions.append({
                "action":   step["name"],
                "step":     step["step"],
                "priority": "high" if deep else "normal",
                "execute":  True
            })

        return actions


# Phase 5.5 — Tool Invocation
# Blueprint: "Memory likhna, Retrieval, Model reasoning, External tool call"
# "AI permission maangti nahi — decide karti hai"
class ToolInvocationEngine:
    def invoke(self, actions: list, tools_layer=None, config: dict = None) -> list:
        """
        Jarvis-only.
        Actions ko actual tool calls se map karna.
        tools_layer inject hoga — tools.py se.
        Abhi structured placeholder — tools.py integration hook ready.
        """
        config = config or {}
        if not config.get("allow_tool_invocation", False):
            return []
        

        results = []

        for action in actions:
            if not action.get("execute", False):
                continue

            action_name = action["action"]

            # Tool mapping — action name → actual tool
            # Future: tools_layer.execute(tool_name, params)
            
            tool_map = {
                "retrieve_knowledge": "knowledge_retrieval",
                "cross_reference":    "scripture_cross_reference",
                "execute":            "code_execution",
                "monitor_result":     "execution_monitor",
                "synthesize":         "reasoning_engine",
                "multi_angle_view":   "hybrid_engine",
            }

            tool_used = tool_map.get(action_name, "llm_reasoning")

            if tools_layer:
                import asyncio
                result = asyncio.run(tools_layer.execute(tool_used, {"query": action_name}))
                status = "done" if result.get("success") else "failed"
            else:
                status = "done"

            results.append({
                "action":    action_name,
                "tool_used": tool_used,
                "status":    status,
                "step":      action.get("step")
            })

        return results


# Phase 5.6 — Execution Monitoring
# Blueprint: "AI dekhe — Action fail hua? Partial success? Unexpected output?"
class ExecutionMonitorEngine:
    def evaluate(self, results: list, config: dict = None) -> dict:
        """
        Jarvis-only.
        Results check karo — success, partial, fail.
        Unexpected output detect karo.
        """
        config = config or {}
        if not config.get("allow_execution_monitoring", False):
            return {"success": False, "status": "disabled", "results": []}
        

        if not results:
            return {"success": False, "status": "no_actions", "results": []}

        failed  = [r for r in results if r.get("status") != "done"]
        success = len(failed) == 0

        return {
            "success":        success,
            "status":         "complete" if success else "partial",
            "failed_actions": failed,
            "results":        results
        }

# =========================
# PHASE 6A: ALIGNMENT FINETUNING
# =========================

class AlignmentFineTuner:
    def evaluate(self, *, question, answer, meta, agency_result, mode):
        if mode != "jarvis":
            return None

        alignment = {
            "clarity_ok": True,
            "safety_ok": True,
            "overreach": False,
            "confidence_level": meta.get("confidence")
        }

        if agency_result and agency_result.get("blocked"):
            alignment["safety_ok"] = True

        if meta.get("confidence", 1.0) < 0.35:    
            alignment["clarity_ok"] = False

        if "must" in answer.lower() or "always" in answer.lower():
            alignment["overreach"] = True

        return alignment
# =========================
# PHASE 6B: KNOWLEDGE FINETUNING
# =========================

class KnowledgeFineTuner:
    def update(self, *, question, world_state, cognitive_profile, alignment_report):
        memory_patch = {
            "question_pattern": question[:120],
            "domain": world_state.get("domain"),
            "used_deep_reasoning": cognitive_profile.get("deep_reasoning"),
            "alignment": alignment_report
        }

        # NOTE:
        # Abhi sirf simulation hai
        # Future me:
        # → vector DB write
        # → model adapters
        # → LoRA / memory graph

        logging.info("[PHASE 6] Knowledge patch stored")
        return memory_patch

class SimplePrompt:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class SimpleParser:
    def parse(self, output) -> str:
        # Ollama / transformers output → string
        return str(output)

class SimpleChain:
    def __init__(self, llm, prompt: SimplePrompt, parser: SimpleParser):
        self.llm = llm
        self.prompt = prompt
        self.parser = parser

    def invoke(self, inputs: dict) -> str:
        formatted_prompt = self.prompt.format(**inputs)

        # LLM call (same as LangChain behavior)
        raw_output = self.llm.invoke(formatted_prompt)

        return self.parser.parse(raw_output)

def create_chain(llm):
    """
    LangChain-free replacement for:
        prompt | llm | StrOutputParser
    Returns a chain object ready to run with context/question at runtime
    """
    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""

    # Prompt object
    prompt = SimplePrompt(template)
    parser = SimpleParser()

    # Chain object
    chain = SimpleChain(llm=llm, prompt=prompt, parser=parser)

    logging.info("Chain created successfully.")
    return chain


# =========================
# PHASE 2.3: META-COGNITION / RETRY ENGINE / SELF-EVALUATION
# =========================

class MetaCognitionEngine:
    def evaluate(
        self,
        answer: str,
        config: dict,
        intent_state: str = None,
        world_state: dict = None,
        cognitive_profile: dict = None
    ):
        """
        Blueprint Phase 2.3 — Light Meta Cognition.
        Answer ki linguistic quality judge karo.
        Rules:
          - NO LLM call
          - No word/sentence/token count
          - spaCy dep_/pos_/morph/ents — language agnostic
          - confidence = float 0.0-1.0 — downstream float ops ke liye
          - Single-pass, max one retry
        """
        doc     = _NLP(answer)
        profile = cognitive_profile or {}
        tokens  = [t for t in doc if not t.is_space and not t.is_punct]

        # ====================================================
        # STEP 1 — LINGUISTIC QUALITY SIGNALS
        # Size se bilkul independent — pure structure/semantics
        # ====================================================

        # Signal 1 — Causal/subordinate reasoning structure
        # dep_ advcl=adverbial clause, mark=subordinator,
        # csubj=clausal subject, relcl=relative clause,
        # xcomp=open clausal complement, ccomp=clausal complement
        # Kisi bhi language mein complex reasoning ka marker
        has_reasoning = any(
            token.dep_ in ["advcl", "mark", "csubj", "relcl", "xcomp", "ccomp"]
            for token in doc
        )

        # Signal 2 — Multi-perspective thinking
        # Multiple nsubj (subjects) + multiple ROOT verbs
        # = system ne ek se zyada angles se socha
        subjects   = [t for t in doc if t.dep_ in ["nsubj", "nsubjpass"]]
        root_verbs = [t for t in doc if t.dep_ == "ROOT"]
        has_multi_perspective = len(subjects) >= 2 and len(root_verbs) >= 2

        # Signal 3 — Factual grounding
        # Named entities present = concrete facts referenced
        # Language agnostic — NER universal hai
        has_factual_grounding = len(doc.ents) >= 1

        # Signal 4 — Structural organization
        # Formatting markers — presence check, not count
        has_structure = any(
            m in answer
            for m in ["1.", "2.", "3.", "- ", "• ", "\n\n", "###", "**"]
        )

        # Signal 5 — Epistemic hedging
        # Conditional mood ya infinitive after modal = uncertainty expressed
        # Positive signal — system ne honest doubt show kiya
        has_hedging = any(
            token.morph.get("Mood") == ["Cnd"]
            or token.morph.get("VerbForm") == ["Inf"]
            for token in doc
            if token.pos_ == "VERB"
        )

        # Signal 6 — Nuanced answer (adversative conjunction)
        # "but/lekin/mais/però/aber" = system ne dono sides dekhi
        has_nuance = any(
            token.dep_ == "cc" and token.pos_ == "CCONJ"
            for token in doc
        )

        # Signal 7 — Incomplete answer
        # Last meaningful token dangling = answer cut off
        # VERB/CCONJ/SCONJ/DET/ADP pe khatam = incomplete
        is_incomplete = (
            bool(tokens) and
            tokens[-1].pos_ in ["VERB", "CCONJ", "SCONJ", "DET", "ADP"]
        ) or answer.strip().endswith("...")

        # Signal 8 — Cognitive investment flags
        deep_used     = profile.get("deep_reasoning", False)
        emergent_used = profile.get("use_emergent_concepts", False)
        parallel_used = profile.get("parallel_thinking", False)

        # ====================================================
        # STEP 2 — SCORE — pure linguistic signals, zero size
        # ====================================================
        score = 0.0

        if has_reasoning:           score += 3.0   # strongest — causal thinking
        if has_multi_perspective:   score += 2.0   # multiple angles
        if has_structure:           score += 1.5   # organized output
        if has_factual_grounding:   score += 1.0   # grounded in facts
        if has_hedging:             score += 1.0   # honest uncertainty
        if has_nuance:              score += 1.0   # both sides considered
        if deep_used:               score += 1.0   # cognitive investment
        if emergent_used:           score += 0.5   # memory enriched
        if parallel_used:           score += 0.5   # parallel angles used
        if is_incomplete:           score -= 4.0   # strong negative

        # ====================================================
        # STEP 3 — CONTEXT-AWARE STANDARD
        # ====================================================
        ethical_weight = (world_state or {}).get("ethical_weight", "low")
        if ethical_weight in ["medium", "high"]:
            score -= 1.0

        # intent_state — Layer 1 computed output — acceptable
        if intent_state == "research":
            score -= 1.0   # research = stricter standard
        if intent_state == "execution":
            score -= 1.0   # execution = accuracy critical

        # ====================================================
        # STEP 4 — CONFIDENCE AS FLOAT 0.0-1.0
        # Downstream float operations ke liye — string nahi
        # MetaRetryEngine, MetaControlEngine, AlignmentFineTuner
        # sab float expect karte hain
        # ====================================================
        max_possible = 11.0   # maximum score possible
        raw_confidence = max(0.0, min(score, max_possible)) / max_possible

        # String label bhi rakho — AlignmentFineTuner ke liye
        if raw_confidence >= 0.6:    confidence_label = "high"
        elif raw_confidence >= 0.35: confidence_label = "medium"
        else:                        confidence_label = "low"

        # ====================================================
        # STEP 5 — RETRY — billing config se
        # ====================================================
        retry_enabled = config.get("meta_retry_enabled", False)
        threshold     = config.get("meta_confidence_threshold", "low")

        retry = False
        if retry_enabled:
            if threshold == "medium" and confidence_label != "high":
                retry = True
            elif threshold == "low" and confidence_label == "low":
                retry = True

        # ====================================================
        # STEP 6 — JARVIS SELF-CRITICAL
        # ====================================================
        self_critical = config.get("meta_self_critical", False)
        if self_critical and confidence_label == "medium" and retry_enabled:
            retry = True

        # ====================================================
        # STEP 7 — INTENT-AWARE OVERRIDE
        # ====================================================
        if retry_enabled and is_incomplete and intent_state in ["research", "execution"]:
            retry = True

        return {
            "confidence":       raw_confidence,      # float — downstream float ops
            "retry":            retry,
            "signals": {
                "has_reasoning":         has_reasoning,
                "has_multi_perspective": has_multi_perspective,
                "has_factual_grounding": has_factual_grounding,
                "has_structure":         has_structure,
                "has_hedging":           has_hedging,
                "has_nuance":            has_nuance,
                "is_incomplete":         is_incomplete,
                "deep_used":             deep_used,
                "emergent_used":         emergent_used,
                "parallel_used":         parallel_used,
                "score":                 score,
                "ethical_weight":        ethical_weight
            }
        }
        

# =========================
# PHASE 2.4: INTENT STATE ENGINE
# =========================

class IntentStateEngine:

    def detect(self, question: str) -> str:
        """
        Phase 1.2 — Multimodal Intent Type Detection
        Blueprint: "Thinking style, not routing. Cross-modal signals (Text + Image + Audio + Doc)"
        Industry: Weighted signal fusion. No English keyword matching.
        Fallback — layer1_bundle available nahi hone par.
        spaCy linguistic signals — language agnostic.
        No English keywords.
        Fallback single-intent detection (legacy / safety)
        """
        if not question or not question.strip():
            return "general"

        doc          = _NLP(question)
        verb_count   = sum(1 for t in doc if t.pos_ == "VERB")
        entity_count = len(doc.ents)
        noun_count   = sum(1 for t in doc if t.pos_ == "NOUN")

        # Causal/subordinate structure → research/analysis intent
        has_causal = any(
            t.dep_ in ["advcl", "mark", "csubj", "relcl"]
            for t in doc
        )

        # High verb count + no entities → process/execution intent
        has_process = verb_count >= 2 and entity_count == 0

        # Named entities + nouns → factual/information intent
        has_factual = entity_count >= 1 and noun_count >= 1

        if has_causal:   return "research"
        if has_process:  return "execution"
        if has_factual:  return "information"
        return "general"

    #-------------------
    # Thinking Styles as defined in Blueprint
    THINKING_STYLES = ["factual", "conceptual", "ethical", "philosophical", "procedural", "mixed"]

    def detect(self, normalized_signal: NormalizedSignal) -> IntentBundle:
        """
        Main entry point.
        Input: NormalizedSignal (from Phase 1.1)
        Output: IntentBundle (Weighted thinking style)
        """
        # 1. Initialize weights for all styles
        weights = {style: 0.0 for style in self.THINKING_STYLES}
        contributing_modalities = []

        # 2. Extract signals from different modalities
        text_signals = normalized_signal.linguistic_signals
        media_meta = normalized_signal.media_metadata
        modality = normalized_signal.modality

        # --- SIGNAL 1: TEXT SEMANTICS (spaCy) ---
        if text_signals:
            contributing_modalities.append("text")
            # Causal/Analytical structure -> Conceptual/Procedural
            if text_signals.get("is_analytical"):
                weights["conceptual"] += 0.4
                weights["procedural"] += 0.2
            
            # Entity heavy -> Factual
            if text_signals.get("has_entity") and text_signals.get("noun_count", 0) > 2:
                weights["factual"] += 0.5
            
            # Verb heavy + no entities -> Procedural
            if text_signals.get("has_verb") and not text_signals.get("has_entity"):
                weights["procedural"] += 0.4

        # --- SIGNAL 2: IMAGE CONTEXT (Objects/Scenes) ---
        if modality == "image" or "image" in modality:
            contributing_modalities.append("image")
            # OCR text analysis for intent
            ocr_text = media_meta.get("ocr_text", "").lower()
            if ocr_text:
                # If OCR contains technical/scientific terms -> Conceptual/Factual
                if any(word in ocr_text for word in ["formula", "diagram", "chart", "structure"]):
                    weights["conceptual"] += 0.3
                    weights["factual"] += 0.2

        # --- SIGNAL 3: AUDIO TONE (Emotion/Urgency) ---
        if modality in ["audio", "voice"]:
            contributing_modalities.append("audio")
            # Use spectral centroid or pitch for tone
            centroid = media_meta.get("spectral_centroid_mean", 0)
            if centroid > 2000: # High brightness/urgency
                weights["ethical"] += 0.2 # Emotional/Urgent often relates to ethics/philosophy
                weights["mixed"] += 0.2

        # --- SIGNAL 4: DOCUMENT TYPE (Structure) ---
        if modality == "document":
            contributing_modalities.append("document")
            # Headings and table count
            if media_meta.get("heading_count", 0) > 3:
                weights["procedural"] += 0.3
                weights["conceptual"] += 0.2
            if media_meta.get("has_tables"):
                weights["factual"] += 0.4

        # 3. Final Fusion & Normalization
        # If multiple weights are high -> 'mixed'
        high_weights = [s for s, w in weights.items() if w >= 0.4]
        if len(high_weights) >= 2:
            weights["mixed"] += sum([weights[s] for s in high_weights])
            # Normalize others
            for s in weights: weights[s] = weights[s] * 0.5

        # Find primary intent
        primary = max(weights, key=weights.get)
        
        # Confidence is the max weight (capped at 1.0)
        confidence = min(1.0, weights[primary])

        return IntentBundle(
            primary_intent=primary,
            intent_weights=weights,
            confidence=confidence,
            cross_modal_signals=contributing_modalities,
            thinking_style=self._get_style_description(primary)
        )

    def _get_style_description(self, style: str) -> str:
        descriptions = {
            "factual": "Direct information retrieval mode",
            "conceptual": "Deep theoretical and abstract thinking",
            "ethical": "Moral and value-based analysis",
            "philosophical": "Existential and metaphysical exploration",
            "procedural": "Step-by-step execution and method logic",
            "mixed": "Multi-dimensional synthesis"
        }
        return descriptions.get(style, "General cognitive mode")

    # Keep detect_from_layer1 as a legacy wrapper for backward compatibility
    def detect_from_layer1(self, layer1_bundle: dict) -> str:
        # This is now a wrapper that calls the new detect() logic
        # In a real flow, we pass the NormalizedSignal object
        return "general" # Placeholder, use detect() instead
    #--------------------

    # ==================================================
    # 🔥 NEW METHOD — LAYER 1 AWARE INTENT DETECTION
    # ==================================================
    def detect_from_layer1(self, layer1_bundle: dict) -> str:
        """
        Layer 1 numeric signals se intent state detect karo.
        No string label match — depth_idx, sub_goals count, domains count.
        """
        if not layer1_bundle:
            return "general"

        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }

        required_depth = layer1_bundle.get("required_depth", "normal")
        depth_idx      = DEPTH_INDEX.get(required_depth, 1)
        sub_goals      = layer1_bundle.get("sub_goals", [])
        domains        = layer1_bundle.get("domains", [])
        normalized_q   = layer1_bundle.get("normalized_query", "")

        # Depth — numeric — strongest signal
        # ultra_deep/very_deep → research (Jarvis bhi cover)
        if depth_idx >= 4:
            return "research"

        # deep + zyada sub_goals → research
        if depth_idx == 3 and len(sub_goals) >= 3:
            return "research"

        # deep + kam sub_goals → analysis
        if depth_idx == 3:
            return "analysis"

        # Cross-domain → research
        if len(domains) >= 3:
            return "research"

        # 2 domains → analysis
        if len(domains) >= 2:
            return "analysis"

        # Complex sub_goals → analysis
        if len(sub_goals) >= 3:
            return "analysis"

        # Sub-goals — spaCy se language agnostic
        for goal in sub_goals:
            if not goal.strip():
                continue
            goal_doc     = _NLP(goal)
            has_causal   = any(
                t.dep_ in ["advcl", "mark", "csubj", "relcl"]
                for t in goal_doc
            )
            verb_count   = sum(1 for t in goal_doc if t.pos_ == "VERB")
            entity_count = len(goal_doc.ents)

            if has_causal:
                return "research"
            if verb_count >= 2 and entity_count == 0:
                return "execution"

        # Fallback — spaCy on normalized query
        if normalized_q:
            return self.detect(normalized_q)

        return "information"


# =========================
# PHASE 2.7: SAFETY / CONSTRAINT LAYER (why & how)
# =========================

class SafetyConstraintEngine:
    def evaluate(self, *, intent, route, config=None, question):
        """
        Blueprint:
          ROLE  : LLM ko call karne se pehle FINAL GATE
          DECIDES: Allow / Block / Reason
          RULES : Intent-Based, Route-Based, Capability-Based (config se)

          Kya NAHI karta:
            - mode string compare nahi — config se
            - Tool mgmt nahi (Phase 5.4)
            - Cognitive load nahi (Phase 2.2)
            - Moral lecture nahi, Emotion nahi, RLHF nahi

          Tier need logic (config-driven):
            Free       → sirf information — curious individual
            Paid       → information + explanation — serious learner
            Ultra Paid → information + analysis — professional individual
            Business   → + research — company ko chahiye
            Enterprise → + research + sensitive domain caution
            Jarvis     → sab allowed — Tony Stark's Jarvis

        Rule-based safety & capability gating
        No RLHF, no emotion, no softness
        """

        config   = config or {}
        decision = {"allow": True, "reason": None}
        #===========================================
                # ===== DANGEROUS KEYWORDS – PUBLIC ONLY =====
        dangerous_keywords = [
            "weapon", "bomb", "explosive", "build missile",
            "chemical synthesis", "harm", "kill"
        ]
        
        # Only block if user is NOT Jarvis (i.e., not allowed dangerous content)
        if not config.get("allow_dangerous_keywords", False):
            if any(k in question.lower() for k in dangerous_keywords):
                decision["allow"] = False
                decision["reason"] = "Potentially dangerous request"
                return decision
        #============================================
        
        # billing config se flags
        allow_research        = config.get("allow_research",        False)
        allow_execution       = config.get("allow_execution",       False)
        allow_agency          = config.get("allow_agency",          False)
        allow_ancient_tech    = config.get("allow_ancient_tech",    False)
        allow_analysis        = config.get("allow_analysis",        False)
        sensitive_domain_flag = config.get("sensitive_domain_caution", False)

        # ====================================================
        # GATE 1 — RESEARCH INTENT
        # Individual tiers ko research nahi — wo researcher nahi hain
        # Business + Enterprise + Jarvis ko chahiye
        # ====================================================
        if intent == "research" and not allow_research:
            decision["allow"] = False
            decision["reason"] = "Research not available at this tier"
            return decision

        # ====================================================
        # GATE 2 — ANALYSIS INTENT
        # Free/Paid ko complex analysis nahi
        # Ultra Paid se milta hai
        # ====================================================
        if intent == "analysis" and not allow_analysis:
            decision["allow"] = False
            decision["reason"] = "Analysis not available at this tier"
            return decision

        # ====================================================
        # GATE 3 — EXECUTION INTENT
        # Sirf Jarvis ko — public tiers mein execution sensitive
        # ====================================================
        if intent == "execution" and not allow_execution:
            decision["allow"] = False
            decision["reason"] = "Execution not available at this tier"
            return decision

        # ====================================================
        # GATE 4 — AGENCY ROUTE
        # Blueprint: "Phase 5 Agency sirf Jarvis ke liye"
        # ====================================================
        if route == "agency" and not allow_agency:
            decision["allow"] = False
            decision["reason"] = "Agency not available at this tier"
            return decision

        # ====================================================
        # GATE 5 — ANCIENT TECH ROUTE
        # Blueprint: "ancient viman/tech misuse prevent"
        # Sirf Jarvis — Viman, Astra, Vedic decode
        # ====================================================
        if route == "ancient_tech" and not allow_ancient_tech:
            decision["allow"] = False
            decision["reason"] = "Ancient technology access not available at this tier"
            return decision

        # ====================================================
        # GATE 6 — SENSITIVE DOMAIN FLAG (Enterprise extra)
        # Large org mein political/legal/financial pe caution
        # Sirf Enterprise mein active — config se
        # ====================================================        
        allow_sensitive_override = config.get("allow_sensitive_override", False)

        if sensitive_domain_flag and route == "sensitive_domain":
            if not allow_sensitive_override:
                decision["allow"] = False
                decision["reason"] = "Sensitive domain restricted at this tier"
                return decision
            # Jarvis mein allow_sensitive_override: True → gate pass hoga

        return decision

# =========================
# PHASE 2.8: TRACE LOGGER
# =========================

class TraceLogger:
    def log(self, trace: dict):
        """
        Internal cognitive trace
        Should NEVER be exposed to public users
        """

        logging.info("===== TRACE LOG =====")
        for k, v in trace.items():
            logging.info(f"{k}: {v}")
        logging.info("=====================")

# =========================
# STEP-1 TRAINING CONFIG                                                 
# =========================       
ENABLE_TRAINING = False   # ❌ abhi OFF Future me ise "True" kr deng jb training krenge
TRAINING_MODE = "offline"  # offline | lora | full_finetune

TRAINING_DATA_DIR = "./training/datasets"
TRAINING_LOG_DIR = "./training/logs"

#=========================================
# STEP-2 UNIVERSAL TRAINING INTERFACE (CORE)
#=========================================
#📌 Is file ko future me kabhi rewrite nahi karna
# 📌 Sirf andar logic upgrade hoga
class TrainingEngine:
    """
    UNIVERSAL TRAINING INTERFACE
    ----------------------------
    - Dataset agnostic
    - Model agnostic
    - Future-proof (LoRA / full finetune)
    """

    def __init__(self, llm, mode="offline"):
        self.llm = llm
        self.mode = mode

    def load_dataset(self, dataset_path):
        """
        Dataset format (JSONL):
        {
          "instruction": "...",
          "input": "...",
          "output": "..."
        }
        """
        

        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def train_step(self, sample):
        """
        SINGLE training interaction
        (later replaced by gradient update)
        """
        prompt = f"""
Instruction:
{sample['instruction']}

Input:
{sample['input']}

Expected Understanding:
{sample['output']}
"""
        return self.llm.invoke(prompt)

    def run_training(self, dataset_path):
        samples = self.load_dataset(dataset_path)

        for idx, sample in enumerate(samples):
            self.train_step(sample)

        return {
            "trained_on_samples": len(samples),
            "mode": self.mode,
            "status": "completed"
        }
      
#==============================================================
#🧱 STEP 3 — PHASE-6 TRAINING CONTROLLER (HOOK)
#==============================================================
class Phase6TrainingController:
    """
    Phase-6 Training Orchestrator
    -----------------------------
    - Controls WHEN training happens
    - Keeps architecture clean
    """

    def __init__(self, training_engine):
        self.engine = training_engine

    def maybe_train(self, dataset_path):
        if not ENABLE_TRAINING:
            return {
                "skipped": True,
                "reason": "Training disabled by config"
            }

        return self.engine.run_training(dataset_path)

# ---- Phase 6C : Training Hook (DISABLED BY DEFAULT) ----
        

training_engine = TrainingEngine(
    llm=llm,
    mode=TRAINING_MODE
)

training_controller = Phase6TrainingController(training_engine)

if ENABLE_TRAINING:
    training_result = training_controller.maybe_train(
        dataset_path="./training/datasets/sample.jsonl"
    )
    logging.info(f"[PHASE 6 TRAINING] → {training_result}")

# =========================================
# LAYER 3 — KNOWLEDGE SOURCE ROUTING (HEART)
# =========================================


# ================================================================
# LAYER 3 — PHASE 3.0: KNOWLEDGE SOURCE CLASSIFIER (UPGRADED)
# Blueprint: "actual mein find karna hai ki kaun si query factual/conceptual"
# 
# Upgrades:
#   1. Registry-based modality signal extraction (no hardcoding)
#   2. Native multimodal signals (mel spectrogram, ViT patches, spatio-temporal)
#   3. Unified embedding space → CrossModalRetrievalHint
#   4. Vedic Symbolic Decoder (Jarvis only, config-gated)
#   5. NLP billing window cap applied
#
# Output: dict (not str) — Phase 3.3 KnowledgeRouter unpacks it
# ================================================================

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# UPGRADE 1: REGISTRY-BASED MODALITY SIGNAL EXTRACTOR
# Blueprint: "Registry-driven, not if/else driven"
# Each modality gets its own handler — Phase 1.1 ke real metadata use karta hai
# ──────────────────────────────────────────────────────────────────

class ModalitySignalRegistry:
    """
    Registry-based dispatch — no hardcoded if/else.
    Naya modality add karna = sirf register() call karo.
    Phase 1.1 ke actual media_metadata keys use karo — averaged scalars nahi.
    """
    _handlers: Dict[str, Any] = {}

    @classmethod
    def register(cls, modality: str):
        """Decorator-based registration — clean, explicit"""
        def decorator(fn):
            cls._handlers[modality] = fn
            return fn
        return decorator

    @classmethod
    def extract(cls, modality: str, media_metadata: dict,
                cognitive_profile: dict) -> dict:
        """
        Dispatch to correct handler.
        Unknown modality → _default handler (safe fallback).
        """
        handler = cls._handlers.get(
            modality,
            cls._handlers.get("_default", lambda m, p: {})
        )
        return handler(media_metadata, cognitive_profile)


# ── AUDIO HANDLER: Log-Mel Spectrogram Signals ─────────────────────
@ModalitySignalRegistry.register("audio")
def _handle_audio(meta: dict, profile: dict) -> dict:
    """
    Native Multimodal: Log-Mel Spectrogram — not just energy.
    Phase 1.1 ke real keys: mel_mean, mel_std, zcr_mean,
    spectral_centroid_mean, tempo_bpm
    
    Vedic insight: Shloka/Mantra = low ZCR + specific mel pattern
    Modern insight: Speech = high ZCR, Music = rhythmic tempo
    """
    mel_mean      = float(meta.get("mel_mean", -40.0))
    mel_std       = float(meta.get("mel_std", 10.0))
    zcr           = float(meta.get("zcr_mean", 0.1))
    centroid_hz   = float(meta.get("spectral_centroid_mean", 2000.0))
    tempo         = float(meta.get("tempo_bpm", 0.0))
    rms           = float(meta.get("rms_before", 0.0))

    # Acoustic complexity — mel_std / |mel_mean| — higher = more tonal variety
    acoustic_complexity = mel_std / max(abs(mel_mean), 1.0)

    # Spectral brightness — normalized to 0-1 (human speech peaks ~2-4kHz)
    spectral_brightness = min(centroid_hz / 8000.0, 1.0)

    # Rhythm density — normalized (fast tempo = high density)
    rhythm_density = min(tempo / 200.0, 1.0)

    # Is Vedic Chant / Mantra:
    # Low ZCR (smooth, sustained) + mel_mean not too low (not silence)
    # + low spectral centroid (bass-dominant, not speech)
    is_chant = (
        zcr < 0.08 and
        mel_mean > -50.0 and
        centroid_hz < 2000.0
    )

    # Is speech: moderate ZCR + high centroid (vocal tract resonance)
    is_speech = zcr > 0.05 and centroid_hz > 1500.0

    # Is music: high tempo OR rhythmic pattern
    is_music = tempo > 60.0 and not is_chant

    return {
        "acoustic_complexity": round(acoustic_complexity, 4),
        "spectral_brightness": round(spectral_brightness, 4),
        "rhythm_density":      round(rhythm_density, 4),
        "is_chant":            is_chant,      # Vedic mantra/shloka
        "is_speech":           is_speech,     # Normal speech query
        "is_music":            is_music,      # Musical content
        "raw_energy":          round(rms, 4),
    }


# ── VOICE HANDLER: Live Voice Stream Signals ───────────────────────
@ModalitySignalRegistry.register("voice")
def _handle_voice(meta: dict, profile: dict) -> dict:
    """
    Voice = live WebRTC stream.
    Phase 1.1 keys: mel_mean, pitch_mean_hz, rms
    Real-time: urgency + pitch variation are key signals.
    """
    pitch_hz  = float(meta.get("pitch_mean_hz", 0.0))
    mel_mean  = float(meta.get("mel_mean", -40.0))
    rms       = float(meta.get("rms", 0.0))

    # Pitch variation — higher pitch = question/urgency (normalized to 1.0 at 500Hz)
    pitch_normalized = min(pitch_hz / 500.0, 1.0)

    # Energy-based urgency
    urgency = min(rms * 5.0, 1.0)

    return {
        "acoustic_complexity": abs(mel_mean) / 80.0,
        "pitch_normalized":    round(pitch_normalized, 4),
        "urgency":             round(urgency, 4),
        "is_chant":            False,   # Voice stream = real-time, not chant
        "is_speech":           True,    # Voice = speech by definition
    }


# ── IMAGE HANDLER: ViT-Style Grid Patch Signals ─────────────────────
@ModalitySignalRegistry.register("image")
def _handle_image(meta: dict, profile: dict) -> dict:
    """
    Native Multimodal: ViT-style patch reasoning proxy.
    Phase 1.1 keys: std_rgb, mean_rgb, ocr_text, original_size
    
    Vedic insight: Stone temples = warm tone, low variance, high edge density
    Grid patches: std per channel ≈ patch richness (how varied each visual region is)
    """
    std_rgb  = meta.get("std_rgb", [0.0, 0.0, 0.0])
    mean_rgb = meta.get("mean_rgb", [128.0, 128.0, 128.0])
    ocr_text = meta.get("ocr_text", "")
    size     = meta.get("normalized_size", [512, 512])

    # Color variance — proxy for visual complexity (ViT patch richness)
    color_var = sum(std_rgb) / len(std_rgb) if std_rgb else 0.0

    # Patch density estimate:
    # High std_rgb = many distinct patch regions (complex scene)
    # Low std_rgb = uniform texture (stone wall, manuscript background)
    patch_density = min(color_var / 80.0, 1.0)

    # Stone/ancient detection:
    # Warm tone (R > B) + low variance (uniform stone texture)
    r_channel = mean_rgb[0] if len(mean_rgb) > 0 else 128
    b_channel = mean_rgb[2] if len(mean_rgb) > 2 else 128
    r_dominance = (r_channel - b_channel) / 255.0

    # Stone: warm (r_dominance > 0.05) + uniform (color_var < 35)
    is_stone_texture = r_dominance > 0.05 and color_var < 35.0

    # Ancient artifact proxy:
    # Stone texture + no OCR (no modern text) OR Sanskrit OCR
    has_ocr       = len(ocr_text.strip()) > 10
    is_likely_ancient = is_stone_texture and (
        not has_ocr or
        any(ch > '\u0900' and ch < '\u097F' for ch in ocr_text)  # Devanagari
    )

    # High-detail image: large size + high patch density
    image_resolution = (size[0] * size[1]) if size else 262144
    is_high_detail    = image_resolution > 500000 and patch_density > 0.4

    return {
        "visual_complexity":   round(min(color_var / 100.0, 1.0), 4),
        "patch_density":       round(patch_density, 4),
        "is_likely_ancient":   is_likely_ancient,    # Temple, yantra, sculpture
        "is_stone_texture":    is_stone_texture,     # Stone carving proxy
        "has_text_regions":    has_ocr,              # OCR found text
        "ocr_confidence":      min(len(ocr_text) / 200.0, 1.0),
        "is_high_detail":      is_high_detail,       # High-res detailed image
    }


# ── VIDEO HANDLER: Spatio-Temporal Cube Signals ─────────────────────
@ModalitySignalRegistry.register("video")
def _handle_video(meta: dict, profile: dict) -> dict:
    """
    Native Multimodal: Spatio-temporal reasoning.
    Phase 1.1 keys: fps, avg_motion, duration_sec, width, height, keyframes_extracted
    
    Vedic insight: Temple walk-through = low motion (static architecture)
    Modern insight: High motion = dynamic scene, procedural content
    """
    fps        = float(meta.get("fps", 25.0))
    avg_motion = float(meta.get("avg_motion", 10.0))
    duration   = float(meta.get("duration_sec", 0.0))
    width      = int(meta.get("width", 640))
    height     = int(meta.get("height", 480))
    keyframes  = meta.get("keyframes_extracted", [])

    # Temporal complexity — high avg_motion = dynamic content (normalized to 1.0 at motion=50)
    temporal_complexity = min(avg_motion / 50.0, 1.0)

    # Spatial richness — high resolution + high fps = dense spatio-temporal cube
    spatial_resolution  = (width * height) / (1920.0 * 1080.0)  # normalize to 1080p
    spatial_richness    = min(spatial_resolution, 1.0)

    # Duration factor — longer = more context to parse
    duration_factor = min(duration / 60.0, 1.0)

    # Architectural content: low motion + multiple keyframes (static structure)
    is_architectural = avg_motion < 8.0 and len(keyframes) >= 3

    # Documentary/lecture: moderate motion + long duration
    is_instructional = avg_motion < 20.0 and duration > 30.0

    return {
        "temporal_complexity":  round(temporal_complexity, 4),
        "spatial_richness":     round(spatial_richness, 4),
        "duration_factor":      round(duration_factor, 4),
        "is_architectural":     is_architectural,    # Temple, ancient structure
        "is_instructional":     is_instructional,    # Lecture, demonstration
    }


# ── DOCUMENT HANDLER: Structural Signals ────────────────────────────
@ModalitySignalRegistry.register("document")
def _handle_document(meta: dict, profile: dict) -> dict:
    """
    Phase 1.1 keys: heading_count, has_tables, page_count, word_count
    """
    heading_count = int(meta.get("heading_count", 0))
    has_tables    = bool(meta.get("has_tables", False))
    page_count    = int(meta.get("page_count", 1))
    word_count    = int(meta.get("word_count", 0))

    # Manuscript: no headings + many pages (continuous Sanskrit text)
    is_manuscript  = heading_count == 0 and page_count > 5

    # Structured report: headings + tables
    is_structured  = heading_count >= 3 and has_tables

    # Structural complexity — normalized
    structural_complexity = min(heading_count / 10.0, 1.0)

    return {
        "structural_complexity": round(structural_complexity, 4),
        "has_tables":            has_tables,
        "page_depth":            min(page_count / 100.0, 1.0),
        "is_manuscript":         is_manuscript,    # Ancient text, scripture
        "is_structured":         is_structured,    # Report, technical doc
    }


# ── DEFAULT HANDLER: Unknown modality safe fallback ─────────────────
@ModalitySignalRegistry.register("_default")
def _handle_default(meta: dict, profile: dict) -> dict:
    return {
        "acoustic_complexity":  0.0,
        "visual_complexity":    0.0,
        "temporal_complexity":  0.0,
        "is_likely_ancient":    False,
        "is_chant":             False,
    }


# ──────────────────────────────────────────────────────────────────
# UPGRADE 2: CROSS-MODAL RETRIEVAL HINT
# Blueprint: "Text query → manuscript image fetch, temple photo → Agni Purana text"
# Unified Embedding Space: query modality ≠ target modality → cross-modal
# ──────────────────────────────────────────────────────────────────

class CrossModalRetrievalHint:
    """
    Unified embedding space hint for Layer 3 retrieval.
    Decides: kya dusri modality ka data bhi chahiye?
    
    Example:
      Audio chant query → retrieve Sanskrit text from Vector DB
      Temple image      → retrieve Agama Shastra text + similar images
      Text query        → standard text retrieval
    """
    def compute(
        self,
        modality: str,
        modality_signals: dict,
        query_embedding: list,
        cognitive_profile: dict
    ) -> dict:
        target_collections = ["text"]  # text hamesha — base

        # Audio/Voice query → also search transcript collection
        if modality in ("audio", "voice"):
            target_collections.append("transcript")

            # Vedic chant → Sanskrit scripture text bhi
            if modality_signals.get("is_chant", False):
                target_collections.append("scripture_text")

        # Image query → OCR text + description text
        if modality == "image":
            if modality_signals.get("has_text_regions", False):
                target_collections.append("ocr_text")
            if modality_signals.get("is_likely_ancient", False):
                # Temple/manuscript → link to Agama Shastra / Vastu text
                target_collections.append("vedic_architecture_text")

        # Video → keyframe descriptions + transcript
        if modality == "video":
            target_collections.append("video_description")
            if modality_signals.get("is_architectural", False):
                target_collections.append("vedic_architecture_text")

        # Document → full text already primary, no extra needed
        cross_modal_active = (
            modality != "text" and
            len(target_collections) > 1
        )

        return {
            "cross_modal_active":     cross_modal_active,
            "target_collections":     target_collections,
            "primary_modality":       modality,
            # Compact embedding hint for Qdrant multi-collection search
            "embedding_hint":         query_embedding[:64] if query_embedding else [],
        }


# ──────────────────────────────────────────────────────────────────
# UPGRADE 3: VEDIC SYMBOLIC DECODER (JARVIS ONLY)
# Blueprint: "Jarvis only — temple carving → mathematical ratios,
#             symbolic/scientific logic"
# Config-gated: capability_level == "jarvis" only
# ──────────────────────────────────────────────────────────────────

class VedicSymbolicDecoder:
    """
    Public tiers: basic OCR/description (Phase 1.1 already does this).
    Jarvis tier: Structural Reasoning → links visuals to Phase-6 trained Vedic blueprints.
    
    Examples:
      Temple pillar image → detect golden ratio → link to Manasara
      Yantra image        → detect geometric proportions → link to Tantra Shastra
      Sanskrit manuscript → symbolic decode → link to Viman Shastra
    
    Brain ko tier pata nahi — config se milta hai.
    """
    # Vedic reference scripture mapping — which visual type → which scripture
    _SCRIPTURE_MAP = {
        "architectural": [
            "Manasara", "Mayamata", "Vastu_Shastra",
            "Agni_Purana", "Brihat_Samhita"
        ],
        "manuscript": [
            "Viman_Shastra", "Samarangana_Sutradhar",
            "Arthashastra", "Sushruta_Samhita"
        ],
        "yantra": [
            "Tantra_Shastra", "Shri_Yantra_Geometry",
            "Atharvaveda", "Devi_Bhagavata"
        ],
        "sculpture": [
            "Shilpa_Shastra", "Vishvakarma_Vastu",
            "Agni_Purana"
        ],
    }

    def decode(self, modality: str, modality_signals: dict,
               cognitive_profile: dict) -> dict:
        """
        Config-gated — brain does not check tier string.
        capability_level == "jarvis" means full power config injected.
        """
        # Config signal — not tier string
        capability_level = cognitive_profile.get("capability_level", "base")
        allow_ancient    = cognitive_profile.get("allow_ancient_tech", False)

        if capability_level != "jarvis" or not allow_ancient:
            # Public tiers: no symbolic decoding
            return {
                "vedic_symbolic_active": False,
                "mode": "basic_description"
            }

        # Only image/video can have symbolic visual content
        if modality not in ("image", "video", "document"):
            return {"vedic_symbolic_active": False, "mode": "non_visual"}

        # Determine visual type from modality signals
        visual_type = "unknown"

        if modality == "image":
            if modality_signals.get("is_likely_ancient"):
                if modality_signals.get("has_text_regions"):
                    visual_type = "manuscript"    # Text visible = manuscript
                elif modality_signals.get("patch_density", 0) > 0.5:
                    visual_type = "yantra"        # High detail = geometric yantra
                else:
                    visual_type = "architectural" # Low detail stone = temple structure

        elif modality == "video":
            if modality_signals.get("is_architectural"):
                visual_type = "architectural"     # Temple walk-through

        elif modality == "document":
            if modality_signals.get("is_manuscript"):
                visual_type = "manuscript"

        # Unknown = basic mode
        if visual_type == "unknown":
            return {"vedic_symbolic_active": False, "mode": "unknown_visual"}

        # Activate symbolic decoding
        scriptures = self._SCRIPTURE_MAP.get(visual_type, [])

        return {
            "vedic_symbolic_active":  True,
            "visual_type":            visual_type,
            "mode":                   "symbolic_decode",
            "cross_reference_scriptures": scriptures,
            "extract_geometric_ratios":   visual_type in ("yantra", "architectural"),
            "extract_scientific_logic":   True,
            # Phase 6 trained blueprint linking — Jarvis memory graph hook
            "link_phase6_blueprints":     True,
        }


# ──────────────────────────────────────────────────────────────────
# PHASE 3.0: ENHANCED KnowledgeSourceClassifier
# Blueprint: "actual mein find karna hai ki kaun si query factual/conceptual"
# Native Multimodal + Registry + Unified Embedding + Vedic Decode
# ──────────────────────────────────────────────────────────────────

class KnowledgeSourceClassifier:
    """
    Phase 3.0 — INDUSTRY UPGRADED
    
    OLD: text-only, averaged scalars, string return
    NEW: registry-based multimodal, rich signals, dict return with all context
    
    Output dict keys:
      category          : "factual" / "conceptual" / "procedural" / "mixed" / "general"
      cross_modal       : CrossModalRetrievalHint dict
      vedic_decode      : VedicSymbolicDecoder dict
      modality_signals  : Registry-extracted rich signals
      nlp_window_used   : billing-gated window size
    """

    def __init__(self):
        self._cross_modal = CrossModalRetrievalHint()
        self._vedic_dec   = VedicSymbolicDecoder()

    def classify(
        self,
        layer1_bundle: dict,
        cognitive_profile: dict = None
    ) -> dict:
        """
        Blueprint: "actual mein find karna hai"
        No English keyword match on user query.
        Pure computed signals: spaCy dep_/pos_, numeric multimodal, embedding scores.
        """
        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }

        profile = cognitive_profile or {}

        # ── STEP A: BILLING NLP WINDOW ────────────────────────────────
        # Blueprint: max_nlp_power from billing controls NLP depth
        nlp_power  = profile.get("max_nlp_power", 5)
        nlp_window = nlp_power * 20  # free=100 tokens, jarvis=2000 tokens

        # ── STEP B: LAYER 1 NUMERIC SIGNALS ──────────────────────────
        required_depth     = layer1_bundle.get("required_depth", "normal")
        depth_idx          = DEPTH_INDEX.get(required_depth, 1)
        normalized_query   = layer1_bundle.get("normalized_query", "")
        sub_goals          = layer1_bundle.get("sub_goals", [])
        is_analytical      = layer1_bundle.get("is_analytical", False)
        has_verb           = layer1_bundle.get("has_verb", False)
        has_entity         = layer1_bundle.get("has_entity", False)
        graph_intent_score = layer1_bundle.get("graph_intent_score", 0.0)
        domains            = layer1_bundle.get("domains", [])
        query_embedding    = layer1_bundle.get("query_embedding", [])
        active_modalities  = layer1_bundle.get("active_modalities", 1)
        thinking_styles    = layer1_bundle.get("thinking_styles_count", 0)

        # ── STEP C: MULTIMODAL METADATA (from Phase 1.1) ─────────────
        media_metadata = layer1_bundle.get("layer1_media_metadata", {})
        modality       = layer1_bundle.get("layer1_modality", "text")

        # ── STEP D: REGISTRY DISPATCH — Native Multimodal Perception ──
        # No if/else — registry handles dispatch
        modality_signals = ModalitySignalRegistry.extract(
            modality, media_metadata, profile
        )

        # ── STEP E: BILLING-GATED NLP ────────────────────────────────
        doc          = _NLP(normalized_query[:nlp_window])
        entity_count = len(doc.ents)
        verb_count   = sum(1 for t in doc if t.pos_ == "VERB")
        noun_count   = sum(1 for t in doc if t.pos_ == "NOUN")

        # Complexity — numeric, Layer 1 computed
        is_complex = len(sub_goals) >= 3 or len(domains) >= 2

        # ── STEP F: CROSS-MODAL RETRIEVAL HINT ───────────────────────
        cross_modal = self._cross_modal.compute(
            modality, modality_signals, query_embedding, profile
        )

        # ── STEP G: VEDIC SYMBOLIC DECODE (Jarvis only) ──────────────
        vedic_decode = self._vedic_dec.decode(
            modality, modality_signals, profile
        )

        # ── STEP H: CLASSIFICATION — Pure Computed Signals ───────────
        # Priority waterfall — no English keywords on user query
        # All decisions: numeric depth, spaCy dep_, embedding scores,
        # registry modality signals

        category = self._classify_category(
            depth_idx=depth_idx,
            is_analytical=is_analytical,
            has_verb=has_verb,
            has_entity=has_entity,
            graph_intent_score=graph_intent_score,
            entity_count=entity_count,
            verb_count=verb_count,
            noun_count=noun_count,
            active_modalities=active_modalities,
            thinking_styles=thinking_styles,
            is_complex=is_complex,
            modality=modality,
            modality_signals=modality_signals,
        )

        result = {
            "category":          category,
            "cross_modal":       cross_modal,
            "vedic_decode":      vedic_decode,
            "modality_signals":  modality_signals,
            "modality":          modality,
            "nlp_window_used":   nlp_window,
            # Pass-through for Phase 3.1+ downstream
            "is_analytical":     is_analytical,
            "graph_intent_score": graph_intent_score,
            "depth_idx":         depth_idx,
        }

        logger.info(
            f"[Ph3.0] category={category} | modality={modality} | "
            f"cross_modal={cross_modal['cross_modal_active']} | "
            f"vedic={vedic_decode['vedic_symbolic_active']} | "
            f"nlp_window={nlp_window} | "
            f"signals_keys={list(modality_signals.keys())}"
        )

        return result

    def _classify_category(
        self, *, depth_idx, is_analytical, has_verb, has_entity,
        graph_intent_score, entity_count, verb_count, noun_count,
        active_modalities, thinking_styles, is_complex,
        modality, modality_signals
    ) -> str:
        """
        Pure classification logic — extracted for testability.
        Priority waterfall: each signal clearly ordered.
        """

        # P1: Multiple modalities = cross-modal = mixed (needs both sources)
        if active_modalities > 1:
            return "mixed"

        # P2: Deep query = conceptual (meaning + relations needed)
        if depth_idx >= 3:
            return "conceptual"

        # P3: Analytical structure (causal dep_) = conceptual
        if is_analytical:
            return "conceptual"

        # P4: Memory Graph strongly activated = conceptual
        if graph_intent_score > 0.6:
            return "conceptual"

        # P5: AUDIO — Vedic chant/mantra = conceptual
        # Native multimodal: mel pattern + ZCR signal (not just energy)
        if modality in ("audio", "voice"):
            if modality_signals.get("is_chant", False):
                return "conceptual"   # Vedic mantra = deep meaning, not factual lookup
            # Normal speech (not chant, not analytical) = factual direction
            if modality_signals.get("is_speech", True) and not is_analytical:
                return "factual"

        # P6: IMAGE — ancient temple/manuscript = mixed
        # ViT patch signals: stone texture + ancient detection
        if modality == "image":
            if modality_signals.get("is_likely_ancient", False):
                return "mixed"   # Structure (factual) + meaning (conceptual)
            if modality_signals.get("visual_complexity", 0) > 0.6 and has_entity:
                return "mixed"

        # P7: VIDEO — architectural walk-through = mixed
        # Spatio-temporal: low motion = static structure
        if modality == "video":
            if modality_signals.get("is_architectural", False):
                return "mixed"

        # P8: DOCUMENT — ancient manuscript = conceptual
        if modality == "document":
            if modality_signals.get("is_manuscript", False):
                return "conceptual"  # Scripture = deep meaning

        # P9: Multiple thinking styles = mixed
        if thinking_styles >= 3:
            return "mixed"

        # P10: Shallow + entity present = factual
        if depth_idx == 0:
            return "factual" if (has_entity or entity_count >= 2) else "general"

        # P11: Complex (many sub-goals or domains) = mixed
        if is_complex:
            return "mixed"

        # P12: Verb-heavy + no entity = procedural (process query)
        if has_verb and not has_entity and verb_count > noun_count:
            return "procedural"

        # P13: Entity present = factual (specific reference)
        if has_entity or entity_count >= 2:
            return "factual"

        # P14: Nouns present = conceptual (abstract topic)
        if noun_count > 0:
            return "conceptual"

        return "general"
# -------- Phase 3.1 : Source Priority Resolution --------
class SourcePriorityResolver:
    def resolve(self, category: str) -> dict:
        if category == "factual":
            # Blueprint: FACTUAL → VECTOR DB only
            # Memory Graph factual questions ke liye nahi — sirf text recall chahiye
            return {"memory": False, "retrieval": True, "reasoning": False}

        if category == "procedural":
            # Process/steps → retrieval se steps milenge + reasoning se execute hoga
            return {"memory": False, "retrieval": True, "reasoning": True}

        if category == "conceptual":
            # Blueprint: CONCEPTUAL → MEMORY GRAPH
            # Meaning + relations chahiye — Memory Graph primary
            return {"memory": True, "retrieval": False, "reasoning": True}

        if category == "mixed":
            # Blueprint: MIXED → BOTH
            return {"memory": True, "retrieval": True, "reasoning": True}

        # general fallback
        return {"memory": True, "retrieval": True, "reasoning": True}


# -------- Phase 3.2 : Confidence Gating --------
class ConfidenceGate:
    def apply(self, routing: dict, memory_score: float, layer1_bundle: dict = None) -> dict:
        if memory_score is None:
            memory_score = 0.5

        bundle        = layer1_bundle or {}
        is_analytical = bundle.get("is_analytical", False)

        # REAL POWER: analytical query ke liye memory band mat karo
        # kyunki analytical query ko Memory Graph chahiye — concepts + relations
        if memory_score < 0.4 and not is_analytical:
            routing["memory"] = False

        return routing
# -------- Phase 3.4 : Ambiguity Detection --------
class AmbiguityDetector:
    def detect(self, layer1_bundle: dict) -> bool:
        """
        Language-agnostic ambiguity detection.
        No hardcoded English strings — spaCy + Layer 1 numeric signals.

        Vague signals:
          - depth shallow     → query surface level hai
          - sub_goals ≤ 1     → Layer 1 kuch decompose nahi kar paya → vague
          - entity_count == 0 → koi specific reference nahi
          - noun_count == 0   → kuch concretely kaha hi nahi

        3+ signals → ambiguous
        """
        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }

        required_depth   = layer1_bundle.get("required_depth", "normal")
        depth_idx        = DEPTH_INDEX.get(required_depth, 1)
        sub_goals        = layer1_bundle.get("sub_goals", [])
        normalized_query = layer1_bundle.get("normalized_query", "")

        # Layer 1 Phase 1.2 signals — real power
        is_analytical      = layer1_bundle.get("is_analytical", False)
        graph_intent_score = layer1_bundle.get("graph_intent_score", 0.0)

        # spaCy — language-agnostic
        doc          = _NLP(normalized_query)
        entity_count = len(doc.ents)
        noun_count   = sum(1 for t in doc if t.pos_ == "NOUN")

        # REAL POWER: analytical query ko ambiguous mat maano
        # causal structure hai → query clear hai, chahe shallow lage
        if is_analytical:
            return False

        # REAL POWER: Memory Graph activated → query ka meaning clear hai
        if graph_intent_score > 0.6:
            return False

        # Numeric vague signals — no English keywords
        is_shallow    = depth_idx <= 1
        few_sub_goals = len(sub_goals) <= 1
        no_entities   = entity_count == 0
        no_nouns      = noun_count == 0

        vague_score = sum([is_shallow, few_sub_goals, no_entities, no_nouns])
        return vague_score >= 3
# -------- Phase 3.5 : Source Conflict Resolution --------
class SourceConflictResolver:
    def resolve(self, routing: dict, layer1_bundle: dict = None) -> dict:
        bundle        = layer1_bundle or {}
        is_analytical = bundle.get("is_analytical", False)
        graph_intent_score = bundle.get("graph_intent_score", 0.0)

        # Blueprint: memory + retrieval dono active → reasoning force
        if routing["memory"] and routing["retrieval"]:
            routing["reasoning"] = True

        # REAL POWER: analytical query → reasoning always on
        if is_analytical:
            routing["reasoning"] = True

        # REAL POWER: graph activated → memory preserve
        if graph_intent_score > 0.6:
            routing["memory"] = True

        domains = bundle.get("domains", [])

        # REAL POWER: multiple domains = cross-domain = dono sources chahiye
        if len(domains) >= 2:
            routing["memory"]    = True
            routing["retrieval"] = True
            routing["reasoning"] = True    

        return routing
    

# -------- Phase 3.6 : Hallucination Guard --------
class HallucinationGuard:
    def apply(self, routing: dict, confidence: float, category: str, layer1_bundle: dict = None) -> dict:
        """
        Factual + low confidence → sirf retrieval.
        Memory aur reasoning band — hallucination risk minimize karo.
        category alag se pass — routing dict mein nahi hota.
        """
        if confidence is None:
            confidence = 0.5

        bundle        = layer1_bundle or {}
        is_analytical = bundle.get("is_analytical", False)

        # Factual + low confidence → sirf retrieval
        # REAL POWER: analytical query pe ye guard nahi lagta
        # kyunki analytical query ko Memory Graph chahiye — reasoning zaroori hai
        if category == "factual" and confidence < 0.3 and not is_analytical:
            routing["memory"]    = False
            routing["reasoning"] = False
            routing["retrieval"] = True

        return routing


# -------- Phase 3.3 : Final Knowledge Router --------
class KnowledgeRouter:
    def route(
        self,
        question: str,
        memory_score: float = 0.5,
        cognitive_profile: dict = None,
        layer1_bundle: dict = None        # Layer 1 bridge — classify ke liye
    ) -> dict:
        """
        Layer 3 ka orchestrator.
        Phase 3.0 → 3.1 → 3.2 → 3.4 → 3.5 → 3.6 sab yahan se chalte hain.
        """
        classifier       = KnowledgeSourceClassifier()
        resolver         = SourcePriorityResolver()
        gate             = ConfidenceGate()
        ambiguity_det    = AmbiguityDetector()
        conflict_res     = SourceConflictResolver()
        hallucination_gd = HallucinationGuard()

        bundle  = layer1_bundle or {}
        profile = cognitive_profile or {}

        # Phase 3.0 — classify (language-agnostic, Layer 1 bundle se)
        category = classifier.classify(bundle, profile)

        # Phase 3.1 — source priority
        routing = resolver.resolve(category)

        # Phase 3.2 — confidence gate
        routing = gate.apply(routing, memory_score, bundle)

        # Phase 3.4 — ambiguity detection (layer1_bundle se — language agnostic)
        is_ambiguous = ambiguity_det.detect(bundle)
        if is_ambiguous:
            routing["retrieval"] = True
            routing["reasoning"] = True

        # Phase 3.5 — source conflict resolution
        routing = conflict_res.resolve(routing, bundle)

        # Phase 3.6 — hallucination guard
        routing = hallucination_gd.apply(routing, memory_score, category, bundle)

        # cognitive_profile — billing gates apply (use_emergent_concepts, deep_reasoning)
        if profile.get("deep_reasoning"):
            routing["reasoning"] = True
        if profile.get("use_emergent_concepts"):
            routing["memory"] = True


        # Layer 1 Phase 1.2 signals — real power
        is_analytical      = bundle.get("is_analytical", False)
        has_entity         = bundle.get("has_entity", False)
        graph_intent_score = bundle.get("graph_intent_score", 0.0)

        # REAL POWER: analytical query → reasoning force
        if is_analytical:
            routing["reasoning"] = True

        # REAL POWER: graph strongly activated → memory force
        if graph_intent_score > 0.6:
            routing["memory"] = True

        # REAL POWER: entity present + not analytical → retrieval boost
        if has_entity and not is_analytical:
            routing["retrieval"] = True
  

        return {
            "use_memory":    routing["memory"],
            "use_retrieval": routing["retrieval"],
            "use_reasoning": routing["reasoning"],
            "category":      category,
            "confidence":    memory_score,
            "is_ambiguous":  is_ambiguous
        }


# ===============================
# Layer 1 : Intent Decomposition Engine
# ===============================

# ─────────────────────────────────────────────
# MODELS — ek baar load, reuse
# spaCy  → linguistic structure (NER, POS, sentences)
# Embedder → semantic meaning (multilingual, no keywords)
# ─────────────────────────────────────────────
try:
    _NLP = spacy.load("xx_ent_wiki_sm")   # multilingual spaCy model
except OSError:
    _NLP = spacy.blank("xx")              # fallback blank

_EMBEDDER = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # 50+ languages natively — Hindi, English, French, Arabic, Japanese — sab
)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class IntentDecompositionEngine:
    """
    Blueprint: "LAYER 1 — Intent ko todna (Control Logic)"
    Blueprint: "intent concept-based hoga, keyword-based nahi"
    Blueprint: "Bina Memory Graph ke Intent sirf text hoga, Meaning nahi"

    Intelligence sources:
    - spaCy  → linguistic structure (entities, POS, sentence boundaries)
    - Embeddings → semantic meaning (any language, no keywords)
    - Memory Graph → hidden goals (concept activation)

    Zero LLM calls. Zero hardcoded keywords.
    """
    # ─────────────────────────────────────────────────────
    def process(self, user_query: str, state: dict,
                config: dict = None, memory_graph=None, **kwargs) -> dict:

        config= config or {}
        
        # ════════════════════════════════════
        # PHASE 1.0 — RAW QUERY CAPTURE (MULTIMODAL + TIER-AWARE)
        # Blueprint: "User ne jo bola exact, bina interpretation"
        # Industry: Real-world signal capture, file uploads, headers, sessions
        # 100% Blueprint Compliant + 100% Production Ready
        # ════════════════════════════════════
        
        # brain/phase1_0_signal_capture_engine.py
        """
        Phase 1.0 — Industry Level Signal Capture Engine
        Blueprint: "User ne jo bola exact, bina interpretation"
        Role: Eyes & Ears only. Zero interpretation. Zero validation.
        Industry: Robust magic-byte detection, multimodal routing, tier-context injection
        """

        
        
        logger = logging.getLogger(__name__)
        
        
        
             

        # Capture user query with all modalities and metadata
        user_email = kwargs.get("user_email", "anonymous@example.com")
        signal_data = get_signal_engine().capture(
        # signal_data = signal_engine.capture(
            raw_input=user_query,
            user_email=user_email,
            content_type=kwargs.get("content_type"),
            headers=kwargs.get("headers"),
            session_id=kwargs.get("session_id"),
            files=kwargs.get("files")
        )
        
        
        
        # Validate request against billing limits
        user_email = kwargs.get('user_email', 'anonymous@example.com')
        validation_result =_VALIDATOR.validate_request(
            signal_data=signal_data,
            user_email=user_email,
            files=kwargs.get('files')
        )
        
        # Check if request is allowed
        if not validation_result.allowed:
            logging.warning(f"[Layer1 Ph1.5] Access denied: {validation_result.violations}")
            # Return error response
            return {
                'error': 'Access denied',
                'violations': validation_result.violations,
                'tier': validation_result.tier,
                'limits_applied': validation_result.limits_applied
            }
        
        # Use validated data
        validated_signal_data = validation_result.filtered_data
        logging.info(f"[Layer1 Ph1.5] Access granted for tier: {validation_result.tier}")
        
        # Update files with validated ones
        if validation_result.filtered_data and hasattr(validation_result.filtered_data, 'metadata'):
            validated_files = validation_result.filtered_data.metadata.get('filtered_files', [])
            if validated_files:
                kwargs['files'] = validated_files
        
        # Extract data for Phase 1.1 - BLUEPRINT COMPLIANT
        # Blueprint: "User ne jo bola exact, bina interpretation"
        # NO MODIFICATION, NO INTERPRETATION
        
        raw_data = validated_signal_data.raw_data  # EXACT CAPTURE - NO MODIFICATION
        signal_type = validated_signal_data.signal_type.value
        query_format = validated_signal_data.query_format.value
        
        # ════════════════════════════════════════════════════════════════
        # PHASE 1.1 — MULTIMODAL NORMALIZATION
        # Blueprint: "Har modality ko processable canonical form mein
        #             convert karna, bina intent distort kiye"
        # ════════════════════════════════════════════════════════════════

        norm_result = _PHASE1_1.normalize(
            raw_data     = raw_data,
            signal_type  = signal_type,
            tier_limits  = validated_signal_data.tier_limits,
            content_type = validated_signal_data.content_type,
        )

        if not norm_result.success:
            logging.warning(
                f"[Layer1 Ph1.1] Normalization partial for {signal_type}: "
                f"{norm_result.error}"
            )

        # Phase 1.1 output → downstream variables
        normalized_query = norm_result.canonical_text
        detected_lang    = norm_result.language

        # spaCy + embedding — Phase 1.2 ke liye
        nlp_window = config.get("max_nlp_power", 5) * 20
        doc = _NLP(normalized_query[:nlp_window]) if normalized_query.strip() else _NLP("")
        query_emb = _EMBEDDER.encode(normalized_query) if normalized_query.strip() \
                    else _EMBEDDER.encode("")

        # Linguistic signals — Phase 1.2 directly yeh use karega
        entities = list(norm_result.linguistic_signals.get("entities", []))
        try:
            noun_chunks = [c.text for c in doc.noun_chunks]
        except Exception:
            noun_chunks = []

        # State mein store — sab downstream phases yahan se padhe
        state["layer1_raw_query"]        = raw_data
        state["layer1_signal_type"]      = signal_type
        state["layer1_query_format"]     = query_format
        state["layer1_capture_id"]       = validated_signal_data.capture_id
        state["layer1_modality"]         = signal_type
        state["layer1_media_metadata"]   = norm_result.media_metadata
        state["layer1_norm_steps"]       = norm_result.processing_steps
        state["layer1_is_media_only"]    = signal_type not in ("text", "code", "data")
        state["layer1_norm_success"]     = norm_result.success
        state["layer1_entities"]         = entities
        state["layer1_noun_chunks"]      = noun_chunks

        logging.info(
            f"[Layer1 Ph1.1] modality={signal_type} | "
            f"lang={detected_lang} | "
            f"steps={norm_result.processing_steps} | "
            f"entities={len(entities)} | "
            f"canonical_len={len(normalized_query)}"
        )
        # media_metadata from norm_result — downstream Phase 1.2 ke liye
        media_payloads = norm_result.media_metadata
        # ════════════════════════════════════════════════════════════════
        # PHASE 1.2 — INTENT TYPE DETECTION (MULTIMODAL THINKING TYPE)
        # Blueprint: "ChatGPT pehle ye decide karta hai — Factual? Conceptual?
        #             Ethical? Philosophical? Procedural? Mixed?
        #             ye routing nahi hai, thinking style hai."
        # Blueprint: "Ab intent sirf text se nahi, cross-modal signals se"
        # All signals: spaCy dep_/pos_ (language-agnostic) + numeric Media metadata
        # Zero English keyword match on user query
        # ════════════════════════════════════════════════════════════════

        # ── spaCy structural signals — language agnostic ──────────────
        verb_count   = sum(1 for t in doc if t.pos_ == "VERB")
        noun_count   = sum(1 for t in doc if t.pos_ == "NOUN")
        entity_count = len(entities)

        is_analytical        = any(t.dep_ in ("advcl", "ccomp", "expl") for t in doc)
        has_verb             = verb_count >= 1
        has_entity           = entity_count >= 1
        has_causal_structure = any(t.dep_ in ("advcl", "mark", "csubj", "relcl") for t in doc)
        has_process_structure = verb_count >= 2 and entity_count == 0
        has_factual_structure = entity_count >= 1 and noun_count >= 1
        has_abstract_nouns   = sum(1 for t in doc if t.pos_ == "NOUN" and len(t.text) > 6) >= 1
        has_imperative       = any(t.pos_ == "VERB" and t.dep_ == "ROOT" for t in doc)
        has_question_struct  = any(t.pos_ == "PRON" for t in doc)
        has_nested_clause    = any(t.dep_ in ("ccomp", "xcomp") for t in doc)
        has_value_judgment   = any(
            t.dep_ in ("advmod", "neg") and t.pos_ == "ADJ" for t in doc
        )

        # ── Cross-modal numeric signals from Phase 1.1 ─────────────────
        # media_payloads = norm_result.media_metadata (set after Phase 1.1)
        # Keys from _normalize_audio: mel_mean, zcr_mean, spectral_centroid_mean
        # Keys from _normalize_image: mean_rgb, std_rgb, ocr_text
        # Keys from _normalize_voice: mel_mean, pitch_mean_hz, rms

        audio_energy    = 0.0   # high energy = urgency/emotion (ZCR + RMS)
        audio_pitch     = 0.0   # pitch mean Hz
        image_edge_density = 0.0  # high edge = complex/structured image
        image_color_variance = 0.0  # high variance = rich/emotional image

        if isinstance(media_payloads, dict):
            # Audio signals — numeric, language agnostic
            zcr  = media_payloads.get("zcr_mean", 0.0)
            rms  = media_payloads.get("rms_before", 0.0)
            audio_energy = float(zcr) * 0.5 + float(rms) * 0.5

            pitch = media_payloads.get("pitch_mean_hz", 0.0)
            audio_pitch = float(pitch) if pitch else 0.0

            # Image signals — numeric, language agnostic
            std_rgb = media_payloads.get("std_rgb", [])
            if std_rgb:
                image_color_variance = float(sum(std_rgb) / len(std_rgb))

        # ── Memory Graph semantic scores — embedding cosine similarity ──
        # Score from Memory Graph = semantic relevance to known concepts
        # We use the raw embedding score directly — no English string match
        semantic_scores = {
            "ethical":       0.0,
            "philosophical": 0.0,
            "procedural":    0.0,
            "factual":       0.0,
            "conceptual":    0.0,
        }

        graph_intent_score = 0.0
        if memory_graph is not None:
            try:
                activated = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=5
                )
                if activated:
                    graph_intent_score = activated[0].get("score", 0.0)

                # Distribute scores by embedding position in concept space
                # Top activated concepts — their weighted scores feed intent types
                # No string matching — pure cosine similarity distribution
                for i, act in enumerate(activated):
                    score  = act.get("weighted_score", act.get("score", 0.0))
                    # Top 1-2 concepts → factual (specific, grounded)
                    # Middle concepts  → conceptual/philosophical (abstract)
                    # Lower concepts   → procedural (process, action)
                    if i == 0:
                        semantic_scores["factual"]       += score * 0.4
                        semantic_scores["conceptual"]    += score * 0.3
                    elif i == 1:
                        semantic_scores["conceptual"]    += score * 0.3
                        semantic_scores["philosophical"] += score * 0.2
                    elif i >= 2:
                        semantic_scores["procedural"]    += score * 0.2
                        semantic_scores["ethical"]       += score * 0.1

                logging.info(
                    f"[Layer1 Ph1.2] graph_score={graph_intent_score:.2f} | "
                    f"semantic={semantic_scores}"
                )
            except Exception as e:
                logging.warning(f"[Layer1 Ph1.2] Memory Graph unavailable: {e}")

        # ── Thinking style indicators — computed signals only ───────────

        # ETHICAL: value judgment structure + high audio energy (emotion)
        # Language agnostic: ADJ+advmod/neg is universal dep_ pattern
        ethical_indicators = [
            semantic_scores["ethical"] > 0.3,
            has_value_judgment,                    # dep_ pattern — any language
            audio_energy > 0.4,                    # numeric — emotion/urgency in voice
        ]
        has_ethical_indicators = any(ethical_indicators)

        # PHILOSOPHICAL: abstract nouns + nested clause + no imperative
        philosophical_indicators = [
            semantic_scores["philosophical"] > 0.3,
            has_abstract_nouns and not has_question_struct and not has_imperative,
            has_nested_clause,                     # ccomp/xcomp — complex thought
        ]
        has_philosophical_indicators = any(philosophical_indicators)

        # PROCEDURAL: process structure + imperative + high image edge density
        procedural_indicators = [
            semantic_scores["procedural"] > 0.3,
            has_process_structure,                 # verb>=2, entity==0
            has_imperative,                        # ROOT VERB — command/instruction
        ]
        has_procedural_indicators = any(procedural_indicators)

        # FACTUAL: entity-heavy + factual structure + no causal
        factual_indicators = [
            semantic_scores["factual"] > 0.3,
            has_factual_structure,                 # entity>=1 + noun>=1
            entity_count >= 1 and not has_causal_structure,
        ]
        has_factual_indicators = any(factual_indicators)

        # CONCEPTUAL: causal structure + abstract + no entities
        conceptual_indicators = [
            semantic_scores["conceptual"] > 0.3,
            has_causal_structure,                  # advcl/mark — "why/because" structure
            noun_count >= 1 and entity_count == 0, # nouns but no specific entities
        ]
        has_conceptual_indicators = any(conceptual_indicators)

        # ── Intent type resolution ──────────────────────────────────────
        intent_type = "general"
        if has_ethical_indicators:
            intent_type = "ethical"
        elif has_philosophical_indicators:
            intent_type = "philosophical"
        elif has_procedural_indicators:
            intent_type = "procedural"
        elif has_factual_indicators and not has_conceptual_indicators:
            intent_type = "factual"
        elif has_conceptual_indicators:
            intent_type = "conceptual"

        # Mixed — multiple thinking styles active
        thinking_styles_detected = sum([
            has_factual_indicators,
            has_conceptual_indicators,
            has_ethical_indicators,
            has_philosophical_indicators,
            has_procedural_indicators,
        ])
        if thinking_styles_detected >= 2:
            intent_type = "mixed"

        # ── Store Phase 1.2 bundle ──────────────────────────────────────
        state["layer1_intent_type"]          = intent_type
        state["layer1_is_analytical"]        = is_analytical
        state["layer1_has_verb"]             = has_verb
        state["layer1_has_entity"]           = has_entity
        state["layer1_graph_intent_score"]   = graph_intent_score
        state["layer1_audio_energy"]         = audio_energy
        state["layer1_image_color_variance"] = image_color_variance
        state["layer1_thinking_styles_count"]= thinking_styles_detected

        logging.info(
            f"[Layer1 Ph1.2] intent_type={intent_type} | "
            f"analytical={is_analytical} | entity={has_entity} | "
            f"causal={has_causal_structure} | imperative={has_imperative} | "
            f"graph_score={graph_intent_score:.2f} | "
            f"audio_energy={audio_energy:.3f} | "
            f"styles_count={thinking_styles_detected}"
        )

        # ════════════════════════════════════
        # PHASE 1.3 — Goal Decomposition
        # Blueprint: "ChatGPT ek question ko multiple invisible goals mein break karta hai"
        # Blueprint: "Bina Memory Graph ke Intent sirf text hoga, Meaning nahi"
        # Sources: Memory Graph (primary) → spaCy noun chunks (fallback)
        # max_goals: config se — billing ka real value
        # No hardcoded strings | No tier names | Language agnostic
        # ════════════════════════════════════

        # config se directly — layer1_config state mein set nahi hota
        max_goals = config.get("max_goals", 2)

        sub_goals = []

        # ── PRIMARY: Memory Graph — semantic hidden goals ──────────────
        # Blueprint: "Bina Memory Graph ke Intent sirf text hoga, Meaning nahi"
        # Memory Graph ke concepts = real hidden goals jo text se nahi milte
        if memory_graph is not None:
            try:
                activated = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=max_goals
                )
                graph_goals = [
                    c["concept"] for c in activated
                    if c.get("score", 0) > 0.3
                ]
                sub_goals.extend(graph_goals)
            except Exception as e:
                logging.warning(f"[Ph1.3] Memory Graph unavailable: {e}")

        # ── SECONDARY: spaCy noun chunks — explicit sub-topics ─────────
        # Blueprint: "Text → noun chunks"
        # noun_chunks already Phase 1.1 se nikle hain — reuse karo
        # Language agnostic — xx_ent_wiki_sm har language mein noun chunks nikalta hai
        if noun_chunks:
            text_goals = [
                chunk for chunk in noun_chunks
                if len(chunk.strip()) > 2
                and chunk.lower() not in sub_goals  # duplicate avoid
            ]
            sub_goals.extend(text_goals)

        # ── TERTIARY: Entities — named anchors as goals ─────────────────
        # Blueprint: "Image → detected objects" proxy
        # spaCy entities = real-world anchors (persons, places, orgs, dates)
        # Language agnostic — NER universal
        if entities:
            entity_goals = [e[0] for e in entities[:max_goals]]
            for eg in entity_goals:
                if eg.lower() not in [g.lower() for g in sub_goals]:
                    sub_goals.append(eg)

        # ── AUDIO signal → depth goal ───────────────────────────────────
        # Blueprint: "Audio → intent emphasis"
        # High audio_energy = urgency — add the core query as priority goal
        if audio_energy > 0.4 and normalized_query not in sub_goals:
            sub_goals.insert(0, normalized_query)  # top priority

        # ── DOCUMENT signal → topic goals ──────────────────────────────
        # Blueprint: "Document → sections/topics"
        # Phase 1.1 media_metadata se real keys: heading_count, sample_headings
        if isinstance(media_payloads, dict):
            headings = media_payloads.get("sample_headings", [])
            for h in headings[:max_goals // 2]:
                if h and h not in sub_goals:
                    sub_goals.append(h)

        # ── Deduplicate + billing limit enforce ─────────────────────────
        seen = set()
        unique_goals = []
        for goal in sub_goals:
            key = goal.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique_goals.append(goal)

        sub_goals = unique_goals[:max_goals]

        logging.info(
            f"[Ph1.3] goals={len(sub_goals)} | max_goals={max_goals} | "
            f"graph_activated={graph_intent_score:.2f} | "
            f"audio_energy={audio_energy:.3f}"
        )

        # ════════════════════════════════════
        # PHASE 1.4 — QUERY EXPANSION
        # Blueprint: "Semantic expansion (embedding neighbors) +
        #             Cross-modal enrichment: Image→tags, Doc→keywords, Audio→emphasis"
        # Output: expanded_queries[]
        # max_query_expansion: config se — billing ka real value
        # Language agnostic — no string match on query
        # ════════════════════════════════════

        max_expansion = config.get("max_query_expansion", 1)

        expanded_queries = [normalized_query]
        temp_expanded    = []

        # ── Primary: Memory Graph semantic expansion ──────────────────
        # Blueprint: "Semantic expansion (embedding neighbors)"
        # Cosine similarity se neighbors — no string match on query
        if memory_graph is not None:
            try:
                neighbors = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=max_expansion
                )
                for n in neighbors:
                    concept = n.get("concept", "")
                    score   = n.get("score", 0.0)
                    # score threshold only — no string match on query
                    if concept and score > 0.3:
                        temp_expanded.append(f"{normalized_query} {concept}")
            except Exception as e:
                logging.warning(f"[Ph1.4] Memory Graph unavailable: {e}")

        # ── Secondary: spaCy entities — linguistic expansion ──────────
        # Blueprint: "Image → object tags add" proxy via NER
        # Entities = real-world anchors — language agnostic
        if entities and len(temp_expanded) < max_expansion:
            entity_texts   = [e[0] for e in entities[:3]]
            entity_variant = f"{normalized_query} {' '.join(entity_texts)}"
            if entity_variant.strip() != normalized_query.strip():
                temp_expanded.append(entity_variant)

        # ── Tertiary: Document headings expansion ─────────────────────
        # Blueprint: "Document → keywords add"
        # Phase 1.1 real key: sample_headings
        if isinstance(media_payloads, dict):
            headings = media_payloads.get("sample_headings", [])
            if headings and len(temp_expanded) < max_expansion:
                heading_variant = f"{normalized_query} {' '.join(headings[:2])}"
                temp_expanded.append(heading_variant)

        # ── Audio: intent emphasis — numeric signal ────────────────────
        # Blueprint: "Audio → intent emphasis"
        # High audio_energy = urgency — sub_goals[0] ko front mein rakh do
        # Duplicate nahi, emphasis: sub_goal concept + query
        if audio_energy > 0.4 and sub_goals:
            emphasis_variant = f"{sub_goals[0]} {normalized_query}"
            if emphasis_variant not in temp_expanded:
                temp_expanded.append(emphasis_variant)

        # ── Final assembly — billing limit enforce ─────────────────────
        for v in temp_expanded:
            if v not in expanded_queries and len(expanded_queries) < max_expansion:
                expanded_queries.append(v)

        logging.info(
            f"[Ph1.4] expanded={len(expanded_queries)} | "
            f"max_expansion={max_expansion} | "
            f"audio_energy={audio_energy:.3f}"
        )

        # ════════════════════════════════════
        # PHASE 1.5 — Reasoning Depth Estimation (COMPLEXITY ENGINE)
        # Blueprint: "ChatGPT decide karta hai — Short Answer chalega ya
        #             heavy multi-layer reasoning chahiye?"
        # Blueprint: "Depth decide hota hai:
        #             (i) Number of modalities
        #             (ii) Conflict between modalities
        #             (iii) Number of goals
        #             (iv) Graph relevance"
        # Output: shallow → ultra_deep
        # Billing cap: config["max_depth"] se enforce
        # ════════════════════════════════════

        n_entities  = len(entities)
        goal_count  = len(sub_goals)

        # ── Signal 1: Graph relevance ─────────────────────────────────
        # Memory Graph se semantic depth — kitna complex concept space hai
        graph_relevance = 0.0
        if memory_graph is not None:
            try:
                graph_relevance = memory_graph.estimate_relevance(normalized_query)
            except Exception:
                pass

        # ── Signal 2: Number of modalities — Blueprint (i) ────────────
        # Har active modality = complexity badhti hai
        # media_payloads dict mein actual modality keys hain
        active_modalities = 1  # text hamesha hota hai
        if isinstance(media_payloads, dict):
            if media_payloads.get("mel_mean") is not None:    # audio/voice
                active_modalities += 1
            if media_payloads.get("mean_rgb") is not None:    # image
                active_modalities += 1
            if media_payloads.get("page_count") is not None:  # document
                active_modalities += 1
            if media_payloads.get("fps") is not None:         # video
                active_modalities += 1

        # ── Signal 3: Conflict between modalities — Blueprint (ii) ────
        # audio high energy + image high variance = contradicting signals
        # thinking_styles_detected >= 3 = multiple cognitive modes active
        has_multimodal_contradiction = False
        if audio_energy > 0.5 and image_color_variance > 50:
            has_multimodal_contradiction = True
        if thinking_styles_detected >= 3:
            has_multimodal_contradiction = True

        # ── Depth decision — all signals combined ──────────────────────
        if has_multimodal_contradiction or active_modalities >= 3:
            required_depth = "ultra_deep"
        elif graph_relevance > 0.8 or goal_count >= 5:
            required_depth = "ultra_deep"
        elif graph_relevance > 0.7 or goal_count >= 4 or active_modalities >= 2:
            required_depth = "very_deep"
        elif goal_count >= 3 or n_entities >= 3:
            required_depth = "deep"
        elif goal_count >= 2 or n_entities >= 2:
            required_depth = "moderate"
        elif goal_count >= 1 or n_entities >= 1:
            required_depth = "normal"
        else:
            required_depth = "shallow"

        # ── Billing cap — config["max_depth"] enforce ─────────────────
        # Brain compute karta hai required_depth
        # Lekin billing decide karta hai max allowed
        depth_levels = ["shallow", "normal", "moderate", "deep", "very_deep", "ultra_deep"]
        max_allowed_depth = config.get("max_depth", "ultra_deep")

        if max_allowed_depth in depth_levels and required_depth in depth_levels:
            if depth_levels.index(required_depth) > depth_levels.index(max_allowed_depth):
                required_depth = max_allowed_depth

        # ── NLP Power Cap — billing se ────────────────────────────────
        # max_nlp_power = kitni deep spaCy processing allowed hai
        # Free=5 (surface), Jarvis=100 (full dependency chain)
        # Yeh signal downstream Phase 1.2 enhanced signals ke liye hai
        # Abhi: token window cap ke roop mein use hoga
        # nlp_power = config.get("max_nlp_power", 5)

        # nlp_power → spaCy processing window (tokens)
        # Free tier: sirf pehle 100 tokens deep analyze
        # Jarvis: poora text analyze
        # Formula: nlp_power * 20 = max tokens spaCy deeply processes
        # nlp_token_window = nlp_power * 20   # free=100, jarvis=2000

        # State mein store — downstream phases use karenge
        state["layer1_nlp_token_window"] = nlp_token_window
        state["layer1_required_depth"]   = required_depth
        state["layer1_active_modalities"]= active_modalities

        logging.info(
            f"[Ph1.5] depth={required_depth} | "
            f"modalities={active_modalities} | "
            f"contradiction={has_multimodal_contradiction} | "
            f"graph={graph_relevance:.2f} | "
            f"goals={goal_count} | entities={n_entities} | "
            f"nlp_window={nlp_token_window}"
        )

        # ════════════════════════════════════════════════════════════════
        # PHASE 1.6 — KNOWLEDGE DOMAIN MAPPING (VERDICT MODERN and CROSS-MODAL)
        # Blueprint: "Scriptures, Science, Ethics, Philosophy, Technology etc"
        # Blueprint: "Domain detect hota hai:
        #             (i) Text semantics (ii) Image category
        #             (iii) Document type (iv) Memory graph"
        # Output: multi-domain tags — numeric signals se, hardcoded labels se nahi
        # ════════════════════════════════════════════════════════════════
        domains = []

        # ── (iv) Memory Graph — primary semantic domain ────────────────
        # Cosine similarity se related concepts → domain emerge karte hain
        # No hardcoded domain names — graph se aate hain
        if memory_graph is not None:
            try:
                activated = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=3
                )
                domains.extend([
                    c["concept"] for c in activated
                    if c.get("score", 0) > 0.35
                ])
            except Exception as e:
                logging.warning(f"[Ph1.6] Memory Graph unavailable: {e}")

        # ── (i) Text semantics — entities + intent type ────────────────
        # spaCy entities = real-world domain anchors (language agnostic)
        if entities:
            domains.extend([ent[0] for ent in entities[:3]])

        # intent_type → domain signal (Phase 1.2 ka output — computed, not label)
        # conceptual/philosophical → abstract domain
        # factual → concrete domain
        # ethical → ethics/dharma domain
        if intent_type in ("philosophical", "ethical"):
            domains.append("dharma_ethics")
        elif intent_type == "procedural":
            domains.append("technical")

        # ── (ii) Image category — numeric signals ─────────────────────
        # Phase 1.1 media_metadata ke real numeric keys se
        # high std_rgb variance = complex/artistic image (visual arts, architecture)
        # low variance + high page_count proxy = technical/diagram
        if isinstance(media_payloads, dict):
            # Image signal — std_rgb variance already computed in Ph1.2
            if image_color_variance > 50:
                domains.append("visual_arts")      # high color variance = rich visual
            elif image_color_variance > 15:
                domains.append("visual_technical")  # moderate = structured/diagram

            # ── (iii) Document type — structural signals ───────────────
            # Phase 1.1 _normalize_document real keys
            has_tables   = media_payloads.get("has_tables", False)
            heading_count = media_payloads.get("heading_count", 0)
            page_count   = media_payloads.get("page_count", 0)

            if has_tables and heading_count > 3:
                domains.append("structured_data")    # tables + headings = report/data
            elif heading_count > 5:
                domains.append("long_form_document")  # many headings = book/paper
            elif page_count > 10:
                domains.append("multi_page_document")

        # ── Analytical signal → science/research domain ────────────────
        # is_analytical already computed in Phase 1.2 — spaCy dep_ signal
        if is_analytical:
            domains.append("analytical")

        # ── Deduplicate + fallback ─────────────────────────────────────
        seen = set()
        unique_domains = []
        for d in domains:
            if d and d.lower() not in seen:
                seen.add(d.lower())
                unique_domains.append(d)

        domains = unique_domains if unique_domains else ["general"]

        logging.info(f"[Ph1.6] domains={domains} | intent_type={intent_type}")

        # ════════════════════════════════════════════════════════════════
        # PHASE 1.7 — INTENT BUNDLE (FINAL MULTIMODAL OBJECT)
        # Blueprint: "LAYER 1 ka output ek object hota hai, jo saari layers use krti"
        # Blueprint: "Ye tumhara AGI ka thought packet hai"
        # Saari next layers isi pe kaam karti hain — sab signals yahan honge
        # ════════════════════════════════════════════════════════════════
        intent_bundle = {
            # ── Core query ──────────────────────────────────────────────
            "raw_query":              raw_data,
            "normalized_query":       normalized_query,
            "query_embedding":        query_emb.tolist(),

            # ── Phase 1.1 — linguistic signals ──────────────────────────
            "entities":               entities,
            "noun_chunks":            noun_chunks,

            # ── Phase 1.2 — thinking type + signals ────────────────────
            "intent_type":            intent_type,
            "is_analytical":          is_analytical,
            "has_verb":               has_verb,
            "has_entity":             has_entity,
            "graph_intent_score":     graph_intent_score,
            "thinking_styles_count":  thinking_styles_detected,

            # ── Phase 1.3 — goals ───────────────────────────────────────
            "sub_goals":              sub_goals,

            # ── Phase 1.4 — expanded queries ────────────────────────────
            "expanded_queries":       expanded_queries,

            # ── Phase 1.5 — depth + modality ────────────────────────────
            "required_depth":         required_depth,
            "active_modalities":      active_modalities,
            "nlp_token_window":       nlp_token_window,

            # ── Phase 1.6 — domains ─────────────────────────────────────
            "domains":                domains,

            # ── Cross-modal numeric signals ─────────────────────────────
            "audio_energy":           audio_energy,
            "image_color_variance":   image_color_variance,
        }

        state["layer1_intent_bundle"] = intent_bundle
        return state
    


#--------LAYER 2 KA PHASE 2.5 MEMORY AWARE QUERY PRUNING (LAYER 4 HOOK)
class MemoryAwareQueryPruner:
    """
    Blueprint: "MEMORY-AWARE QUERY PRUNING (LAYER 4 HOOK) —
                Layer 4 se check hota hai:
                (i) kya ye knowledge already store hai?
                (ii) kya repeat query waste hoti?
                RESULT: (i) drop (ii) merge (iii) deeper"

    Drop   = score > drop_threshold — tier-aware (max_docs se derive hota hai)
    Merge  = duplicate strings — seen set se deduplicate (language agnostic)
    Deeper = score low + required_depth deep — naya topic, rakhna zaroori
    Koi string label nahi, koi hardcoded English nahi.
    """

    def prune(
        self,
        queries: list,
        memory_graph,
        layer1_bundle: dict,
        cognitive_profile: dict = None    # ADDED — billing signals ke liye
    ) -> list:
        """
        queries          : Phase 2.4 ka output — plain string list
        memory_graph     : Layer 4 hook — estimate_relevance() call
        layer1_bundle    : Layer 1 signals — required_depth, is_analytical, etc.
        cognitive_profile: Billing config — max_docs, deep_reasoning

        Returns: pruned list — drop/merge/deeper decisions applied
        """
        # memory_graph None ho to pruning skip — safe fallback
        if memory_graph is None or not queries:
            return queries

        profile = cognitive_profile or {}

        # ── BILLING: drop threshold — max_docs se derive karo ────────
        # max_docs bada = zyada resources allowed = lenient drop (Jarvis)
        # max_docs chota = resource constrained = aggressive drop (free)
        # Formula: 0.75 + log10(max_docs+1) / 10.0 capped at 0.97
        # free(6)=0.83, paid(12)=0.86, ultra(25)=0.89, biz(35)=0.91,
        # enterprise(50)=0.92, jarvis(999999)=0.97
        import math
        max_docs = profile.get("max_docs", 6)
        if max_docs == -1 or max_docs >= 999999:
            drop_threshold = 0.97   # jarvis — almost never drop
        else:
            drop_threshold = min(0.97, 0.75 + math.log10(max_docs + 1) / 10.0)

        # ── BILLING: deep_reasoning → lenient threshold ───────────────
        # deep_reasoning=True = system deeply explore karna chahta hai
        # → threshold raise karo (zyada queries rakhne do)
        deep_reasoning = profile.get("deep_reasoning", False)
        if deep_reasoning:
            drop_threshold = min(0.97, drop_threshold + 0.03)

        # ── Layer 1 signals ────────────────────────────────────────────
        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }
        bundle               = layer1_bundle or {}
        required_depth       = bundle.get("required_depth", "shallow")
        depth_idx            = DEPTH_INDEX.get(required_depth, 0)
        is_deep              = depth_idx >= 3

        # Layer 1 Phase 1.2 signals — real power
        is_analytical        = bundle.get("is_analytical", False)
        graph_intent_score   = bundle.get("graph_intent_score", 0.0)
        audio_energy         = bundle.get("audio_energy", 0.0)        # ADDED
        active_modalities    = bundle.get("active_modalities", 1)     # ADDED
        thinking_styles      = bundle.get("thinking_styles_count", 0) # ADDED

        # ── Multimodal keep signals ────────────────────────────────────
        # audio_energy > 0.4 = urgent voice query — kabhi drop nahi
        # active_modalities > 1 = complex multimodal input — context zaroori
        # thinking_styles >= 2 = multi-angle query — keep karo
        is_multimodal_complex = (
            audio_energy > 0.4 or
            active_modalities > 1 or
            thinking_styles >= 2
        )

        pruned  = []
        seen    = set()
        dropped = 0
        kept    = 0

        for q in queries:
            # ── MERGE — duplicate deduplicate ─────────────────────────
            # Blueprint: "kuch Merge"
            # lowercase dedup — language agnostic
            key = q.lower().strip()
            if key in seen:
                continue
            seen.add(key)

            try:
                score = memory_graph.estimate_relevance(q)
            except Exception as e:
                logging.warning(f"[Ph2.5] memory error: {e}")
                pruned.append(q)
                kept += 1
                continue

            # ── DROP — strongly in memory, repeat waste hogi ──────────
            # Blueprint: "kya ye knowledge already store hai? drop karo"
            # Guard: analytical + multimodal complex queries NEVER drop
            if score > drop_threshold and not is_analytical and not is_multimodal_complex:
                dropped += 1
                continue

            # ── DEEPER — naya topic, rakhna zaroori ───────────────────
            # Blueprint: "score low + required_depth deep = naya topic, rakhna zaroori"
            if is_deep and score < 0.3:
                pruned.append(q)
                kept += 1
                continue

            # ── GRAPH ACTIVATED ────────────────────────────────────────
            # graph_intent_score > 0.6 = concepts graph mein strongly connected
            if graph_intent_score > 0.6 and score > 0.3:
                pruned.append(q)
                kept += 1
                continue

            # ── NORMAL — rakhna hai ────────────────────────────────────
            pruned.append(q)
            kept += 1

        logging.info(
            f"[Ph2.5] in={len(queries)} kept={kept} dropped={dropped} "
            f"threshold={drop_threshold:.2f} is_deep={is_deep} "
            f"audio={audio_energy:.2f} modalities={active_modalities}"
        )

        # Blueprint: fallback — kuch to dena hai Layer 3 ko
        return pruned if pruned else queries

#================================================
#==========LAYER 2 : ADAPTIVE QUERY EXPANSION(DYNAMICS)==========
#================================================
# .......Phase 2.1 : INTENT-wise QUERY BRANCHING.........
class IntentQueryBrancher:
    """
    Blueprint: "har intent ke liye alag query path banana"
    Industry Upgrade: Multimodal branching —
      Text: sub_goals + entities + noun_chunks
      Audio: audio_energy se urgency branch
      Image: image_color_variance se visual branch
      Document: sample_headings se section branches
    Language agnostic — no string match on query content.
    """

    def branch(
        self,
        base_query: str,
        layer1_bundle: dict,
        cognitive_profile: dict = None
    ) -> list:

        # ── Layer 1 signals ────────────────────────────────────────────
        sub_goals      = layer1_bundle.get("sub_goals", [])
        entities       = [e[0] for e in layer1_bundle.get("entities", [])]
        noun_chunks    = layer1_bundle.get("noun_chunks", [])
        required_depth = layer1_bundle.get("required_depth", "normal")
        is_analytical  = layer1_bundle.get("is_analytical", False)
        has_verb       = layer1_bundle.get("has_verb", False)
        has_entity     = layer1_bundle.get("has_entity", False)
        graph_score    = layer1_bundle.get("graph_intent_score", 0.0)

        # ── Multimodal signals from Layer 1 ───────────────────────────
        audio_energy         = layer1_bundle.get("audio_energy", 0.0)
        image_color_variance = layer1_bundle.get("image_color_variance", 0.0)
        active_modalities    = layer1_bundle.get("active_modalities", 1)
        # document headings — Phase 1.3 ne sub_goals mein daal diye already
        # sample_headings directly bundle mein nahi — sub_goals se aate hain

        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }
        depth_idx = DEPTH_INDEX.get(required_depth, 1)

        profile = cognitive_profile or {}
        COMPLEXITY_BRANCHES = {
            "low": 1, "normal": 2, "high": 3,
            "very_high": 4, "expert": 4, "maximum": 5
        }
        max_branches = COMPLEXITY_BRANCHES.get(
            profile.get("query_complexity", "normal"), 2
        )

        # Multimodal: zyada modalities = zyada branches
        if active_modalities >= 3:
            max_branches = min(max_branches + 2, 7)
        elif active_modalities >= 2:
            max_branches = min(max_branches + 1, 6)

        branches = []
        seen     = set()

        def add(q: str):
            # Dedup by seen set — no cross-language string match on content
            key = q.strip()
            key_norm = key.lower()
            if key and key_norm not in seen:
                seen.add(key_norm)
                branches.append(q)

        # ── Branch 1: Base query — hamesha ────────────────────────────
        add(base_query)

        # ── Branch 2: sub_goals — Memory Graph concepts ────────────────
        # No string match on base_query — seen set handles dedup
        for goal in sub_goals[:max_branches]:
            add(f"{base_query} {goal}")

        # ── Branch 3: Entities — named anchors ────────────────────────
        if entities:
            add(f"{base_query} {' '.join(entities[:max_branches])}")

        # ── Branch 4: noun_chunks — structural topics ──────────────────
        for chunk in noun_chunks[:max_branches]:
            add(f"{chunk} {base_query}")

        # ── Branch 5: Deep query — entity-first angle ─────────────────
        if depth_idx >= 3:
            for ent in entities[:2]:
                add(f"{ent} {base_query}")

        # ── Branch 6: Analytical — causal angle ───────────────────────
        if is_analytical and graph_score > 0.3:
            for goal in sub_goals[:2]:
                add(f"{goal} {base_query}")

        # ── Branch 7: Procedural — verb-first angle ───────────────────
        if has_verb and not has_entity and noun_chunks:
            add(f"{noun_chunks[0]} {base_query}")

        # ── INDUSTRY UPGRADE: Audio modality branch ───────────────────
        # Blueprint industry: "Audio → intent emphasis"
        # High audio_energy = urgency → narrow/direct branch
        if audio_energy > 0.4 and sub_goals:
            add(f"{sub_goals[0]}")      # most important goal alone
            add(f"{base_query}")        # already added but reinforces urgency

        # ── INDUSTRY UPGRADE: Image modality branch ───────────────────
        # High color variance = rich visual → visual description branch
        # Low variance = structured/technical → precise branch
        # ── INDUSTRY: Image modality branch ───────────────────────────
        # image_color_variance numeric — language agnostic
        # High variance = rich/artistic → Memory Graph goals se context
        # Low-medium variance = structured → entity-anchored precise
        # No hardcoded English strings — sirf numeric + real signals
        if image_color_variance > 50 and sub_goals:
            # Rich image: top goals Memory Graph se combine karo
            for goal in sub_goals[:2]:
                add(f"{goal} {base_query}")
        elif image_color_variance > 15 and entities:
            # Structured image: entity-anchored precise branch
            add(f"{entities[0]} {base_query}")

        return branches
#.............. Phase 2.2 : QUERY GRANULARITY DECISION ..........
class QueryGranularityDecider:
    """
    Blueprint: "HAR INTENT KE LIYE DECIDE HOTA HAI —
                Narrow: exact facts, Broad: Landscape, Abstract: Philosophy"
    Narrow/Broad/Abstract — 0.0-1.0 scope_score — labels nahi, pure number.
    Industry Upgrade: audio_energy + image_color_variance + active_modalities
                      + thinking_styles_count scope mein factor hote hain.
    cognitive_profile: billing se — deep_reasoning, use_emergent_concepts.
    """

    def decide(
        self,
        branches: list,
        layer1_bundle: dict,
        memory_graph,
        cognitive_profile: dict = None   # ADDED — billing access ke liye
    ) -> list:

        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }

        # ── Layer 1 text signals ───────────────────────────────────────
        required_depth     = layer1_bundle.get("required_depth", "shallow")
        depth_idx          = DEPTH_INDEX.get(required_depth, 0)
        depth_score        = depth_idx / (len(DEPTH_INDEX) - 1)  # 0.0-1.0
        is_analytical      = layer1_bundle.get("is_analytical", False)
        has_entity         = layer1_bundle.get("has_entity", False)
        has_verb           = layer1_bundle.get("has_verb", False)
        graph_intent_score = layer1_bundle.get("graph_intent_score", 0.0)

        # ── Layer 1 multimodal signals ─────────────────────────────────
        audio_energy         = layer1_bundle.get("audio_energy", 0.0)
        image_color_variance = layer1_bundle.get("image_color_variance", 0.0)
        active_modalities    = layer1_bundle.get("active_modalities", 1)
        thinking_styles      = layer1_bundle.get("thinking_styles_count", 0)

        # ── Billing signals ────────────────────────────────────────────
        profile        = cognitive_profile or {}
        deep_reasoning = profile.get("deep_reasoning", False)
        use_emergent   = profile.get("use_emergent_concepts", False)

        # ── Intent-based scope bias — spaCy computed signals ──────────
        # is_analytical → abstract scope (causal structure detected)
        # has_entity only → narrow scope (specific facts needed)
        # has_verb only → broad scope (process/landscape)
        if is_analytical:
            intent_bias = 0.3
        elif has_entity and not is_analytical:
            intent_bias = -0.2
        elif has_verb and not has_entity:
            intent_bias = 0.1
        else:
            intent_bias = 0.0

        # ── Memory Graph bias ──────────────────────────────────────────
        graph_bias = graph_intent_score * 0.2

        # ── INDUSTRY: Multimodal scope signals ─────────────────────────
        # High audio_energy = urgency → narrow/focused scope needed
        # Blueprint industry: urgent audio queries need direct retrieval
        audio_bias = -0.1 if audio_energy > 0.4 else 0.0

        # High image_color_variance = rich/complex visual → broader scope
        image_bias = 0.1 if image_color_variance > 50 else (
                     0.05 if image_color_variance > 15 else 0.0)

        # Multiple modalities = more context needed = broader scope
        modality_bias = min((active_modalities - 1) * 0.05, 0.15)

        # Multiple thinking styles = complex query = broader scope
        thinking_bias = min(thinking_styles * 0.03, 0.1)

        # ── Billing scope adjustments ──────────────────────────────────
        # deep_reasoning = billing says go deeper = push toward abstract
        deep_bias = 0.1 if deep_reasoning else 0.0
        # use_emergent = billing says use graph = push toward abstract
        emergent_bias = 0.05 if use_emergent else 0.0

        results = []
        for branch in branches:
            mem_score = 0.0
            if memory_graph is not None:
                try:
                    mem_score = memory_graph.estimate_relevance(branch)
                except Exception as e:
                    logging.warning(f"[Ph2.2] memory error: {e}")

            # scope_score — all signals combined
            # depth 40% + memory 25% + intent 20% + multimodal 10% + billing 5%
            scope_score = round(
                (depth_score   * 0.40) +
                (mem_score     * 0.25) +
                intent_bias          +
                graph_bias           +
                audio_bias           +
                image_bias           +
                modality_bias        +
                thinking_bias        +
                deep_bias            +
                emergent_bias,
                4
            )
            # 0.0–1.0 clamp
            scope_score = max(0.0, min(1.0, scope_score))
            results.append({"branch": branch, "scope_score": scope_score})

        logging.info(
            f"[Ph2.2] branches={len(results)} | "
            f"depth_score={depth_score:.2f} | intent_bias={intent_bias:.2f} | "
            f"audio_bias={audio_bias:.2f} | image_bias={image_bias:.2f} | "
            f"modality_bias={modality_bias:.2f} | thinking_bias={thinking_bias:.2f} | "
            f"deep_bias={deep_bias:.2f}"
        )
        return results

#.................. Phase 2.3 : DYNAMIC QUERY SHAPE GENERATOR ..........
class QueryShapeGenerator:
    """
    Blueprint: "yahaan actually query forms bante hai —
                Declaration, Exploratory, Hypothetical, Comparative, Causal.
                ye phase random nhi hota, decision-based hota hai."
    Decision = Phase 2.2 ka scope_score + Layer 1 signals.
    Forms = Memory Graph ke close concepts se emerge karte hain.
    Language agnostic — no string match on content.
    Industry Upgrade: audio_energy (Hypothetical), image_color_variance (Exploratory).
    """
    def generate(
        self,
        branch_item: dict,
        query_embedding: list,
        memory_graph,
        cognitive_profile: dict = None,
        layer1_bundle: dict = None
    ) -> str:
        branch      = branch_item["branch"]
        scope_score = branch_item["scope_score"]

        # Billing gate — free tier: no enrichment
        if not (cognitive_profile or {}).get("use_emergent_concepts", False):
            return branch

        # Narrow query — enrichment noise banega
        if scope_score < 0.2 or memory_graph is None or not query_embedding:
            return branch

        # Layer 1 signals — real power
        bundle             = layer1_bundle or {}
        is_analytical      = bundle.get("is_analytical", False)
        has_verb           = bundle.get("has_verb", False)
        has_entity         = bundle.get("has_entity", False)
        graph_intent_score = bundle.get("graph_intent_score", 0.0)
        audio_energy       = bundle.get("audio_energy", 0.0)
        image_color_variance = bundle.get("image_color_variance", 0.0)
        thinking_styles    = bundle.get("thinking_styles_count", 0)

        try:
            top_k = max(1, round(scope_score * 3))
            similar = memory_graph.get_similar_concepts(query_embedding, top_k=top_k)

            # Score threshold only — no string match on content
            # Hindi/Sanskrit/Arabic concepts — language agnostic
            close_concepts = [
                c["concept"] for c in similar
                if c.get("score", 0) > 0.5
            ]

            if not close_concepts:
                return branch

            # ── CAUSAL shape ─────────────────────────────────────────
            # Blueprint: "Causal (Why/How)"
            # Trigger: is_analytical — spaCy dep_ advcl/mark structure
            # Real power: causal query → concept ko causal angle se link
            if is_analytical:
                return f"{branch} {close_concepts[0]}"

            # ── COMPARATIVE shape ─────────────────────────────────────
            # Blueprint: "Comparative"
            # Trigger: graph strongly activated → related concepts compare
            # Real power: Memory Graph score > 0.6, 2+ concepts available
            if graph_intent_score > 0.6 and len(close_concepts) >= 2:
                return f"{branch} {close_concepts[0]} {close_concepts[1]}"

            # ── HYPOTHETICAL shape ────────────────────────────────────
            # Blueprint: "Hypothetical"
            # Trigger: audio_energy high = urgency/speculation OR
            #          multiple thinking styles = complex/uncertain query
            # Real power: top 2 concepts combine — speculative angle
            # Industry: voice queries often hypothetical ("what if...")
            if audio_energy > 0.4 or thinking_styles >= 2:
                if len(close_concepts) >= 2:
                    return f"{branch} {close_concepts[0]} {close_concepts[1]}"
                return f"{branch} {close_concepts[0]}"

            # ── EXPLORATORY shape ─────────────────────────────────────
            # Blueprint: "Exploratory"
            # Trigger: high scope OR rich image (visual complexity)
            # Real power: multiple concepts → wide exploration
            if scope_score > 0.7 or image_color_variance > 50:
                return f"{branch} {' '.join(close_concepts[:3])}"

            # ── DECLARATION shape ─────────────────────────────────────
            # Blueprint: "Declaration"
            # Trigger: entity present + narrow scope = definitive statement
            # Real power: entity-anchored single concept
            if has_entity and scope_score < 0.5:
                return f"{branch} {close_concepts[0]}"

            # ── PROCEDURAL shape (verb-first) ─────────────────────────
            # Not in blueprint explicitly but maps to process queries
            # Trigger: verb heavy, no entity
            if has_verb and not has_entity:
                return f"{branch} {close_concepts[0]}"

            # Default
            return f"{branch} {' '.join(close_concepts)}"

        except Exception as e:
            logging.warning(f"[Ph2.3] memory error: {e}")

        return branch
 
#...............Phase 2.4 : ABSTRACTION LEVEL MODULATOR ..........
class AbstractionModulator:
    """
    Blueprint: "Same intent ko multiple abstract levels pr query krta hai —
                Concrete (facts,data), Conceptual (models,theories), Meta (ethics,philosophy)"
    Phase 2.3 ne close concepts liye (concrete form).
    Phase 2.4 graph distance se abstraction level decide karta hai.
    Industry Upgrade: audio_energy (Concrete prefer), image_color_variance (Conceptual).
    Language agnostic — sirf score ranges, no string match.
    """

    def adjust(
        self,
        query: str,
        scope_score: float,
        query_embedding: list,
        memory_graph,
        cognitive_profile: dict = None,
        layer1_bundle: dict = None
    ) -> str:

        # Billing gate — free tier: no abstraction expansion
        if not (cognitive_profile or {}).get("use_emergent_concepts", False):
            return query

        if scope_score < 0.5 or memory_graph is None or not query_embedding:
            return query

        # Layer 1 signals
        bundle               = layer1_bundle or {}
        is_analytical        = bundle.get("is_analytical", False)
        has_entity           = bundle.get("has_entity", False)
        audio_energy         = bundle.get("audio_energy", 0.0)
        image_color_variance = bundle.get("image_color_variance", 0.0)

        try:
            similar = memory_graph.get_similar_concepts(query_embedding, top_k=8)

            # ── CONCRETE level ────────────────────────────────────────
            # Blueprint: "Concrete (facts, data)"
            # Trigger: entity present + audio urgent + not analytical
            # → High-score concepts (score > 0.7) add — entity anchored
            # Industry: urgent audio = user needs direct facts, not philosophy
            if has_entity and not is_analytical and (scope_score < 0.7 or audio_energy > 0.4):
                concrete = [
                    c["concept"] for c in similar
                    if c.get("score", 0) > 0.7  # score threshold only — no string match
                ]
                if concrete:
                    return f"{query} {concrete[0]}"
                return query

            # ── CONCEPTUAL level ───────────────────────────────────────
            # Blueprint: "Conceptual (models, theories)"
            # Trigger: scope 0.65-0.8 OR rich image (complex visual = conceptual thinking)
            # → Medium-distance concepts (0.3-0.5 score range)
            # Industry: rich image = spatial/structural concept needed
            if 0.65 <= scope_score <= 0.8 or image_color_variance > 50:
                conceptual = [
                    c["concept"] for c in similar
                    if 0.3 <= c.get("score", 0) <= 0.5
                ]
                if conceptual:
                    return f"{query} {conceptual[0]}"

            # ── META level ─────────────────────────────────────────────
            # Blueprint: "Meta (ethics, philosophy)"
            # Trigger: scope > 0.8 OR is_analytical (deep causal reasoning)
            # → Distant concepts (0.15-0.3 score range) — philosophy/ethics
            if scope_score > 0.8 or is_analytical:
                meta = [
                    c["concept"] for c in similar
                    if 0.15 < c.get("score", 0) < 0.3
                ]
                if meta:
                    return f"{query} {meta[0]}"

        except Exception as e:
            logging.warning(f"[Ph2.4] memory error: {e}")

        return query




#..........Phase 2.6 : QUERY PRIORITY & BUDGET ALLOCATION ..........
class QueryBudgetAllocator:
    """
    Blueprint: "sab Question equal nhi hote —
                Kaun Phle? Kaun Shallow? Kaun Deep?
                Factors: intent importance, cognitive budget"

    Priority = Memory Graph se measure — inverted relevance score
    Reason: jo query Memory Graph mein kam known = naya topic = pehle explore karo
            jo zyada known = already covered = baad mein ya drop
    Budget  = cognitive_profile.max_docs — billing se aata hai
    Koi string label nahi, koi hardcoded English nahi.
    """

    def allocate(
        self,
        queries: list,
        cognitive_profile: dict,
        memory_graph,
        layer1_bundle: dict = None
    ) -> list:
        """
        queries          : Phase 2.5 ka pruned output
        cognitive_profile: billing se aaya — max_docs budget hai
        memory_graph     : Layer 4 hook — priority measure ke liye
        layer1_bundle    : Layer 1 signals — intent importance ke liye
        Returns: budget ke andar top-priority queries — sorted
        """
        # Empty fallback — Layer 3 ko crash nahi hona chahiye
        if not queries:
            return queries

        budget = cognitive_profile.get("max_docs", 6)

        # Layer 1 signals — real power
        bundle               = layer1_bundle or {}
        is_analytical        = bundle.get("is_analytical", False)
        graph_intent_score   = bundle.get("graph_intent_score", 0.0)
        audio_energy         = bundle.get("audio_energy", 0.0)        # ADDED
        active_modalities    = bundle.get("active_modalities", 1)     # ADDED
        thinking_styles      = bundle.get("thinking_styles_count", 0) # ADDED

        # Billing signals
        deep_reasoning = cognitive_profile.get("deep_reasoning", False)

        def priority_score(q: str) -> float:
            base = 0.5

            # Memory Graph — primary priority signal
            # Invert: low memory = naya topic = HIGH priority (explore pehle)
            # Blueprint: "Kaun Pehle?" = unexplored pehle
            if memory_graph is not None:
                try:
                    mem_score = memory_graph.estimate_relevance(q)
                    base = 1.0 - mem_score
                except Exception as e:
                    logging.warning(f"[Ph2.6] memory error: {e}")

            # Blueprint: "intent importance" — analytical query = deeper intent
            # REAL POWER: spaCy dep_ signal — language agnostic
            if is_analytical:
                base += 0.2

            # Memory Graph strongly activated — graph-related queries pehle
            if graph_intent_score > 0.6:
                base += 0.15

            # deep_reasoning = billing says explore deeply
            if deep_reasoning:
                base += 0.1

            # ADDED: audio_energy > 0.4 = urgent voice query = highest priority
            # Blueprint: "intent importance" — urgency = pehle answer do
            if audio_energy > 0.4:
                base += 0.25

            # ADDED: active_modalities > 1 = complex multimodal = zyada priority
            # More modalities = richer context = explore pehle
            if active_modalities > 1:
                base += 0.15

            # ADDED: thinking_styles >= 2 = multi-angle complex query = priority boost
            if thinking_styles >= 2:
                base += 0.1

            return min(base, 1.0)

        prioritized = sorted(queries, key=priority_score, reverse=True)
        logging.info(
            f"[Ph2.6] budget={budget} total={len(queries)} "
            f"selected={min(budget, len(queries))} "
            f"audio={audio_energy:.2f} modalities={active_modalities}"
        )
        return prioritized[:budget]
#............Phase 2.7 : FINAL QUERY BUNDLE OUTPUT..........
class AdaptiveQueryExpansionEngine:
    """
    Blueprint: "final clean output jo layer 3 ko milega —
                ab layer 3 blind search nhi karta, wo intelligent routing karta hai"

    Phase 2.7 = orchestrator — Phase 2.1 se 2.6 tak sab chalata hai.
    Final output: clean list of strings — Layer 3 ko milega.
    Koi extra cutting nahi — Phase 2.6 ne already budget enforce kar di.
    Koi hardcoded English nahi.
    """

    def run(self, question: str, layer1_bundle: dict, cognitive_profile: dict, memory_graph) -> list:
        brancher            = IntentQueryBrancher()
        granularity_decider = QueryGranularityDecider()
        shape_gen           = QueryShapeGenerator()
        abstraction         = AbstractionModulator()
        allocator           = QueryBudgetAllocator()

        # Phase 2.1 — branches
        branches = brancher.branch(
            layer1_bundle.get("normalized_query", question),
            layer1_bundle,
            cognitive_profile
        )

        # Phase 2.2 — scope scores
        query_scope_items = granularity_decider.decide(
            branches, layer1_bundle, memory_graph, cognitive_profile
        )

        # Phase 2.3 + 2.4 — shape + abstraction
        query_emb = layer1_bundle.get("query_embedding")
        if query_emb is None:
            logging.warning("[Ph2.7] query_embedding missing from layer1_bundle — Ph2.3/2.4 skipping Memory Graph")

        queries = []
        for item in query_scope_items:
            q = shape_gen.generate(item, query_emb, memory_graph, cognitive_profile, layer1_bundle)
            q = abstraction.adjust(q, item["scope_score"], query_emb, memory_graph, cognitive_profile, layer1_bundle)
            queries.append(q)

        # Guard — agar queries empty ho gayi (edge case)
        if not queries:
            logging.warning("[Ph2.7] queries empty after Ph2.3/2.4 — falling back to normalized_query")
            queries = [layer1_bundle.get("normalized_query", question)]

        # Phase 2.5 — prune
        queries = MemoryAwareQueryPruner().prune(queries, memory_graph, layer1_bundle, cognitive_profile)

        # Phase 2.6 — priority + budget
        queries = allocator.allocate(queries, cognitive_profile, memory_graph, layer1_bundle)

        # Ph 2.8 Trace — billing flag se control (sirf Jarvis dev mode mein ON)
        if cognitive_profile.get("trace_logging", False):
            logging.info(f"[TRACE Ph2.1-branches] count={len(branches)}")
            logging.info(f"[TRACE Ph2.2-scope_items] count={len(query_scope_items)}")
            logging.info(f"[TRACE Ph2.3-shape] first_query={queries[0] if queries else 'empty'}")
            logging.info(f"[TRACE Ph2.4-abstraction] query_emb_present={query_emb is not None}")
            logging.info(f"[TRACE Ph2.6-budget] max_docs={cognitive_profile.get('max_docs', 6)}")
            logging.info(f"[TRACE Ph2.7-final_count] {len(queries)}")

        logging.info(f"[Ph2.7] Final query count → {len(queries)}")
        return queries
#===========================================================
#==========LAYER 5 : REASONING & SYNTHESIS =================
#===========================================================

#............Phase 5.1 - Evidence Aggregation Engine........
class EvidenceAggregator:
    def collect(
        self,
        retrieved_docs=None,
        memory=None,
        reasoning_output=None,
        world_state=None,
        agency_result=None
    ):
        return {
            "retrieval": retrieved_docs or [],
            "memory": memory,
            "reasoning": reasoning_output,
            "world_state": world_state,
            "agency": agency_result
        }

#............Phase 5.2 - Contradiction Detection Engine...............
class ContradictionDetector:
    def detect(self, evidence_bundle):
        conflicts = []

        retrieval = evidence_bundle.get("retrieval", [])
        memory = evidence_bundle.get("memory")

        if memory and retrieval:
            for doc in retrieval:
                if memory in doc.page_content:
                    continue
                conflicts.append({
                    "type": "memory_vs_retrieval",
                    "detail": "Memory and retrieved docs mismatch"
                })

        return {
            "has_conflict": len(conflicts) > 0,
            "conflicts": conflicts
        }

#................Phase 5.3 - Source Trust Scoring............
class SourceTrustScorer:
    def score(self, evidence_bundle, cognitive_profile, route=None ):
        score = {
            "memory": 0.7,
            "retrieval": 0.7,
            "reasoning": 0.8
        }

        if cognitive_profile.get("deep_reasoning"):
            score["reasoning"] += 0.1

        if cognitive_profile.get("confidence", 0.6) < 0.6:
            score["retrieval"] += 0.1

        if route == "memory":
            score["memory"] += 0.1
        elif route == "retrieval":
            score["retrieval"] += 0.1
    
        
        return score

#................Phase 5.4 - Reconcillation Engine...................
class ReconciliationEngine:
    def reconcile(self, evidence_bundle, conflicts, trust_scores):
        if not conflicts["has_conflict"]:
            return evidence_bundle["reasoning"]

        # Conflict hai → weighted decision
        if trust_scores["reasoning"] >= trust_scores["retrieval"]:
            return evidence_bundle["reasoning"]

        return "Re-evaluated answer based on stronger evidence"

#.............Phase 5.5 - Answer Structuring Engine..................
class AnswerStructurer:
    def structure(self, resolved_answer, intent_state):
        if intent_state == "research":
            return f"""
### Summary
{resolved_answer}

### Key Points
- Evidence-backed reasoning
- Conflicts resolved
- Sources reconciled
"""
        return resolved_answer

#................Phase 5.6 - Self-Judgement Engine.........
class SelfJudgeEngine:
    def evaluate(self, answer, cognitive_profile):
        score = 0.7
        if cognitive_profile.get("deep_reasoning"):
            score += 0.1

        return {
            "score": score,
            "acceptable": score >= 0.75
        }

#................Phase 5.7 - Final Synthesis Engine.............
class FinalSynthesisEngine:
    def synthesize(
        self,
        evidence_bundle,
        conflicts,
        trust_scores,
        intent_state,
        cognitive_profile
    ):
        reconciler = ReconciliationEngine()
        base_answer = reconciler.reconcile(
            evidence_bundle, conflicts, trust_scores
        )

        # parallel_thinking — multiple reasoning angles se synthesize karo
        if cognitive_profile.get("parallel_thinking"):
            angles = []

            # Angle 1 — retrieval-first
            retrieval_heavy = dict(evidence_bundle)
            retrieval_heavy["reasoning"] = ""
            angle1 = reconciler.reconcile(retrieval_heavy, conflicts, trust_scores)
            if angle1:
                angles.append(f"[Evidence-Based View]\n{angle1}")

            # Angle 2 — reasoning-first
            reasoning_heavy = dict(evidence_bundle)
            reasoning_heavy["retrieval"] = []
            angle2 = reconciler.reconcile(reasoning_heavy, conflicts, trust_scores)
            if angle2:
                angles.append(f"[Reasoning-Based View]\n{angle2}")

            # Angle 3 — memory context
            if evidence_bundle.get("memory"):
                angles.append(f"[Memory Context]\n{evidence_bundle['memory']}")

            if angles:
                base_answer = base_answer + "\n\n" + "\n\n".join(angles)

            logging.info(f"[Parallel Thinking] {len(angles)} angles synthesized")

        structurer = AnswerStructurer()
        structured = structurer.structure(base_answer, intent_state)

        judge = SelfJudgeEngine()
        judgement = judge.evaluate(structured, cognitive_profile)

        return structured, judgement
    
# #----------------------------------------------------
# # =========================
# # SECTION 7 — OUTPUT BOUNDARY & RESPONSE GUARD
# # =========================

# class OutputBoundaryGuard:
#     """
#     FINAL RESPONSE BOUNDARY
#     -----------------------
#     - Runs AFTER Layer-5 synthesis
#     - Does NOT modify reasoning logic
#     - Only enforces output safety + policy
#     """

#     def enforce(self, *, answer: str, mode: str, intent: str, cognitive_profile: dict = None) -> str:
#         final_answer = answer.strip()

#         # ---- Public hard limits ----
#         if mode == "public":
#             # ❌ No commands / authority tone
#             forbidden_phrases = [
#                 "you must",
#                 "you should always",
#                 "it is mandatory",
#                 "guaranteed",
#                 "100%"
#             ]
#             for p in forbidden_phrases:
#                 if p in final_answer.lower():
#                     final_answer = final_answer.replace(p, "")
#                     final_answer += f"\n\n[Boundary Notice: authority tone adjusted for public mode]"


        
#             # ❌ No excessive certainty
#             if "always" in final_answer.lower():
#                 final_answer += "\n\n(Note: This may vary depending on context.)"
        



#         # ---- Jarvis mode transparency ----
#         if mode == "jarvis" and intent == "research":
#             final_answer += "\n\n— Generated with extended reasoning enabled."

#         return final_answer

# # =========================
# # SECTION 8 — DEPLOYMENT GOVERNANCE
# # =========================

# class DeploymentGovernor:
#     """
#     Deployment & Tier Governance
#     ----------------------------
#     - Controls exposure
#     - NOT part of cognition
#     """

#     def apply(self, *, mode: str, intent: str, response: str) -> str:
#         # ---- Public tier constraints ----
#         if mode == "public":
#             if intent in ["research", "execution"]:
#                 response = (
#                     "[Public Notice] Response adjusted for public tier visibility.\n\n"
#                     + response
#                 )

#         response = (
#             f"[Public Tier: {PUBLIC_TIER.upper()}]\n\n"
#             + response
#         )


#         # ---- Jarvis audit tag ----
#         if mode == "jarvis":
#             response += "\n\n[Governance: Jarvis-tier execution]"

#         return response


   
#========================================================================================================================
#============================================================================================================================
    
async def main(
    question: str,
    config: Dict[str, Any],
    vector_db,                     # injected
    memory_graph: MemoryGraph,     # injected
    conversation_memory: ConversationMemory,  # injected
    conversation_id: str = None
) -> str:
    """  
    AGI Brain - Pure Intelligence Layer  
      
    Args:  
        question: User query  
        config: Tier configuration from BillingLayer  
            - tier: str (free/paid/ultra_paid/business_small/enterprise/jarvis)  
            - max_docs: int  
            - deep_reasoning: bool  
            - use_emergent_concepts: bool  
            - collection: str  
            - allowed_tools: list  
            - max_tokens: int  
    """  
    # Extract mode from config  
    # tier = config["tier"]  
    


    training_result = training_controller.maybe_train(
        dataset_path="./training/datasets/sample.jsonl"
    )

    logging.info(f"[PHASE 6 TRAINING] → {training_result}")

    # Initialize the language model
          

    chain = create_chain(llm)

    # question = "How to report The Prince ?"=======================================

    memory_layer = conversation_memory 
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())   # sirf fallback  
     

    # Layer 4 Full Memory Graph (Blueprint ka REAL BRAIN - Persistent + Tier-aware)
    memory_graph_full = memory_graph
    await memory_graph_full.init_connections(memory_scope=config.get("memory_scope", "public_only"))

    # Layer 8 + Layer 4 combined initialization complete

     # Layer 8 connections initialize (async)
    await memory_layer.init_connections()

    # ================================
    # LAYER 1 : INTENT DECOMPOSITION
    # ================================
    
    intent_engine = IntentDecompositionEngine()
    
    state = {}
    state = intent_engine.process(
        user_query=question,
        state=state,
        config=config,
        memory_graph=memory_graph_full   # Layer 4 hook — Blueprint aligned
    )
    layer1_bundle = state.get("layer1_intent_bundle", {})
    # state["layer1_intent_bundle"] now contains:
    # - intent_type
    # - sub_goals
    # - expanded_queries
    # - domains
    # - required_depth
    #----------------------------------------------------------------

    # ===== Phase 2.4 : Intent State =====
    intent_engine = IntentStateEngine()
    intent_state = intent_engine.detect_from_layer1(layer1_bundle)
    
    logging.info(f"[Intent State] → {intent_state}")

    # ================================
    # LAYER 2A — COGNITIVE ROUTER
    # ================================
    
    router = CognitiveRouter()
    cognitive_route = router.route_with_context(
        question=question,
        domains=layer1_bundle.get("domains", []),
        required_depth=layer1_bundle.get("required_depth", "normal"),
        layer1_bundle=layer1_bundle,    # ← numeric signals ke liye
        cognitive_profile=cognitive_profile    # Jarvis bypass ke liye
    )
   
    logging.info(f"[Cognitive Route] → {cognitive_route}")

    #-------------------------------------------------------------
    world_state = {
        "domain": layer1_bundle.get("domains", ["general"])[0],
        # Layer 1 ka domains list — agar koi bhi domain detect hua = human context possible
        "human_factor": len(layer1_bundle.get("domains", [])) > 0,
        # required_depth se decide — string label nahi
        # "ethical_weight": "high" if layer1_bundle.get("required_depth") in ["deep", "very_deep", "ultra_deep"] else "low",
        "ethical_weight": "high" if {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }.get(layer1_bundle.get("required_depth", "normal"), 1) >= 3 else "low"
    }
    
    # world_state = {
    #     "domain": layer1_bundle.get("domains", ["general"])[0],
    #     "human_factor": "ethics" in layer1_bundle.get("domains", []),
    #     "ethical_weight": "high" if layer1_bundle.get("intent_type") in ["ethical", "philosophical"] else "low"
    # }
   
    # ================================
    # LAYER 2B — COGNITIVE LOAD CONTROLLER
    # ================================
    
    load_controller = CognitiveLoadController()
    
    cognitive_profile = load_controller.decide(
        route=cognitive_route,
        config=config,
        world_state=world_state,
        intent=intent_state,
        required_depth=layer1_bundle.get("required_depth", "normal")
        
    )
    # ===== 🔒 LAYER-2 HARD GUARANTEES (ADD THIS) =====
    cognitive_profile.setdefault("deep_reasoning", False)
    cognitive_profile.setdefault("use_emergent_concepts", False)
    cognitive_profile.setdefault("max_docs", config.get("max_docs", 6))
    cognitive_profile.setdefault("query_complexity", "normal")

    # ===== Layer-2 Confidence Bootstrap =====
    cognitive_profile.setdefault(
        "confidence",
        0.6 if cognitive_profile.get("deep_reasoning") else 0.75
    )
    logging.info(f"[Cognitive Profile] → {cognitive_profile}")
    # Ab system ko pata hai :
          #1. kitna deep sochna hai | 2.fast/ slow/ research mode | 3.token + reasoning budget | layer 3 isi output pr chalegi

 
      
    
    
    # ================================
    # LAYER 2 — ADAPTIVE QUERY EXPANSION
    # ================================
    
    adaptive_query_engine = AdaptiveQueryExpansionEngine()
    
    adaptive_queries = adaptive_query_engine.run(
        question=question,
        layer1_bundle=layer1_bundle,
        cognitive_profile=cognitive_profile,
        memory_graph=memory_graph_full   # Layer 4 hook
    )
    
    # 🔒 Layer-2 Production Lock
    if not isinstance(adaptive_queries, (dict, list)):                     #parmanent
        raise RuntimeError("Layer-2 output corrupted. Blueprint violation.")

    
    logging.info(f"[Adaptive Queries] → {adaptive_queries}")
         

    # ===== Layer-2 → Router Bridge (Blueprint Compliant Fix) =====
    adaptive_query_text = " ".join(adaptive_queries) if adaptive_queries else ""
    logging.info(f"[Layer2→Layer3] Combined Query → {adaptive_query_text}")
    query_count = len(adaptive_queries)
    logging.info(f"[Layer2] final query_count={query_count}")

    # ================================
    # LAYER 3A — WORLD MODEL
    # ================================
    
    world_engine = WorldModelEngine()
    
    world_state = world_engine.analyze(
        question=question,
        intent=intent_state,
        cognitive_profile=cognitive_profile
    )
    # ===== Layer-2 → Layer-3 Cognitive Contract =====
    world_state["cognitive_confidence"] = cognitive_profile.get("confidence", 0.6)
    world_state["reasoning_depth"] = cognitive_profile.get("deep_reasoning", False)
    world_state["allowed_uncertainty"] = (
        "high" if cognitive_profile.get("deep_reasoning") else "medium"
    )

    logging.info(f"[World Model] → {world_state}")
    
    
    # ================================
    # LAYER 3B — WORLD ASSUMPTIONS
    # ================================
    
    assumption_engine = WorldAssumptionEngine()
    
    world_state = assumption_engine.enrich(
        world_state,
        domains=layer1_bundle.get("domains", []),
        assumption_checking=cognitive_profile.get("assumption_checking", False)
    )
    
    logging.info(f"[World Assumptions] → {world_state}")
    
    
    # ================================
    # LAYER 3C — QUERY MUTATION
    # ================================
    
    query_mutator = WorldQueryMutator()
    
    mutated_question = query_mutator.mutate(
        question,
        world_state,
        expanded_queries=layer1_bundle.get("expanded_queries", [])
    )
    
    logging.info(f"[Mutated Question] → {mutated_question}")
     
    # ab system janta hai :
                       #1. hidden assumptions | 2.missing context | 3. user ne jo bola,wo bhi  
    # ──────────────── Layer 8: Fetch previous conversation history ────────────────
    history = await memory_layer.get_conversation_history(
        conversation_id=conversation_id,
        email=config.get("email", None)  # email se tier khud nikalti hai
    )

    # History ko current context mein mix kar do (very useful for continuity)
    if history.strip():
        mutated_question = f"Previous conversation context:\n{history}\n\nCurrent question: {mutated_question}"

    # Optional: History ko mutated_question mein mix kar sakte ho
    # mutated_question = f"{history}\n\nCurrent query: {mutated_question}"

 
    # ===== Phase 3.3 : FINAL KNOWLEDGE ROUTING (KnowledgeRouter) =====
    router3 = KnowledgeRouter()
    knowledge_route = router3.route(
        question=(
            mutated_question + " " +
            " ".join(layer1_bundle.get("expanded_queries", [])) + " " + adaptive_query_text # 👈 VERY IMPORTANT
        ),
        memory_score=memory_graph_full.estimate_relevance(question),  # actual memory relevance
        cognitive_profile=cognitive_profile,
        layer1_bundle=layer1_bundle                                    # Layer 1 bridge
    )
    
    final_route = (
        "memory" if knowledge_route["use_memory"]
        else "retrieval" if knowledge_route["use_retrieval"]
        else "reasoning"
    )
   
    
    
    
    # routing_decision decides:
    # - use_memory
    # - use_retrieval
    # - use_reasoning
    
    
    
    

    

    logging.info(f"[Router Decision] → {final_route}")

    # ===== Vedic + Ancient-Modern Blend — Jarvis only (Blueprint: Layer 4 Memory Graph hook) =====
    # Blueprint: "Mercury Vortex Engine → Ion Thrusters connect karna"
    # Blueprint: "Ancient + Modern blend karke new inventions"
    if cognitive_profile.get("vedic_cross_reference"):
        mutated_question = (
            mutated_question
            + "\n\n[Vedic Cross-Reference Active]"
            + " Cross-reference with: Rigveda, Atharvaveda, Upanishads, Arthashastra,"
            + " Viman Shastra, Samarangana Sutradhar, Sushruta Samhita, Chanakya Niti."
            + " Find scientific logic hidden in Sanskrit descriptions."
        )
        logging.info("[Vedic Cross-Reference] Activated — scripture context injected into query")

    if cognitive_profile.get("ancient_modern_blend"):
        mutated_question = (
            mutated_question
            + "\n\n[Ancient-Modern Blend Active]"
            + " Map ancient Indian concepts to modern science:"
            + " Prana = Bioelectric field, Vimana = Aerodynamic craft,"
            + " Chakra = Nerve plexus, Yagna = Thermodynamic process,"
            + " Mantra = Cymatics/Sound frequency, Ayas = Advanced metallurgy."
            + " Do not dismiss ancient knowledge — decode its engineering basis."
            + " Propose modern reconstruction where possible."
        )
        logging.info("[Ancient-Modern Blend] Activated — blending instructions injected into query")
    # ===== Phase 2.5 : Response Strategy =====
    strategy_engine = ResponseStrategyEngine()
    response_strategy = strategy_engine.decide(
        route=final_route,
        intent={
            "state": intent_state,
            "required_depth": layer1_bundle.get("required_depth", "normal")
        },
        config=config,
        cognitive_profile=cognitive_profile
    )

    logging.info(f"[Response Strategy] → {response_strategy}")


    # if mode == "jarvis":
    #     cognitive_profile["deep_reasoning"] = True
    #     cognitive_profile["use_emergent_concepts"] = True
    #     cognitive_profile["max_docs"] = 15


    
    # if SYSTEM_MODE == "jarvis" and intent_state == "research":
    
    #     cognitive_profile.update({
    #     "deep_reasoning": True,
    #     "max_docs": 12
    #     })

    
     # ===== Phase 2.7 : Safety / Constraint =====
    safety_engine = SafetyConstraintEngine()
    safety = safety_engine.evaluate(
        intent=intent_state,
        route=final_route,
        config=config,
        question=question
    )

    logging.info(f"[Safety Check] → {safety}")

    if not safety["allow"]:
        print(f"Request blocked: {safety['reason']}")
        return

    
                
        
        
    
    # ======= Routed Cognition ========    
    if final_route == "memory":
        memory = memory_lookup(vector_db, question, cognitive_profile=cognitive_profile)
        if memory:
            final_prompt = f"""
Use the following memory context to answer carefully:

{memory}

Question:
{question}
"""
            res = llm_generate(final_prompt)
        else:
            res = "No relevant memory found."

    elif final_route == "retrieval":

        docs = []
        memory = None
    

            # Phase 3 complete ho chuka
        
        # ===== Phase 4.0 =====
        self_state_engine = SelfStateEngine()
        self_state = self_state_engine.build(
            config=config,
            intent=intent_state,
            route=final_route,
            world_state=world_state
        )
        
        # ===== Phase 4.1 =====
        cap_engine = CapabilityAwarenessEngine()
        self_state = cap_engine.evaluate(self_state)
        
        # ===== Phase 4.2 =====
        confidence_engine = SelfConfidenceEngine()
        self_state = confidence_engine.estimate(self_state, world_state)
        
        # ===== Phase 4.3 =====
        refinement_engine = SelfRefinementEngine()
        cognitive_profile = refinement_engine.decide(
            self_state,
            cognitive_profile
        )
        
        # ===== Phase 4.4 : Heavy Meta-Cognition =====
        heavy_meta_engine = HeavyMetaCognitionEngine()
        cognitive_profile, meta_flags = heavy_meta_engine.evaluate(
            question=question,
            self_state=self_state,
            world_state=world_state,
            cognitive_profile=cognitive_profile
        )
        
        logging.info(f"[Heavy Meta] → {meta_flags}")

        

        # ============================
        # Phase 4.5 — Meta-Control Loop
        # ============================
        
        meta_control = MetaControlEngine()
        
        decision = meta_control.decide(
            question=question,
            cognitive_profile=cognitive_profile,
            meta_flags=meta_flags,
            self_state=self_state
        )
        
        # Apply meta-level override if required
        if decision.override:
            cognitive_profile = decision.updated_profile   
                
                    
        
        
        # ==================================
        # Phase 4.6 — Meta-Failure / Retry
        # ==================================
        
        meta_retry = MetaRetryEngine()
        
        retry_signal = meta_retry.check(
            cognitive_profile=cognitive_profile,
            self_state=self_state
        )
        
        # Adjust cognition if retry is required
        if retry_signal.should_retry:
            cognitive_profile = meta_retry.adjust(cognitive_profile)

        

        # 🔁 Phase 4 effect propagation
        max_docs = cognitive_profile["max_docs"]
        use_emergent = cognitive_profile.setdefault("use_emergent_concepts", False)
        deep_reasoning = cognitive_profile.setdefault("deep_reasoning", False)

        

        

        # ====== Phase 2 : Implicit Memory =====
        docs, emergent_concepts = implicit_memory_retrieval(
            vector_db,
            mutated_question,
            cognitive_profile=cognitive_profile   # billing ka max_docs respect hoga
        )
        
        # ===== Phase 3.3 : Conflict Detection =====
        conflict_detector = ConflictDetector()
        conflict_signals = conflict_detector.analyze(docs)

        if conflict_signals["conflict"] and not cognitive_profile.get("deep_reasoning"):
            cognitive_profile["deep_reasoning"] = True


        # if conflict_signals["conflict"]:
        #     cognitive_profile["deep_reasoning"] = True


        # 🔹 Phase 3.2: World–Memory Binding
        binder = WorldMemoryBinder()
        emergent_concepts = binder.bind(world_state, emergent_concepts)

        
        # ===== Phase 3.5 : Emergent Concept Injection =====
        if cognitive_profile.get("use_emergent_concepts") and emergent_concepts:
            mutated_question = (
                mutated_question
                + " "
                + " ".join(emergent_concepts[:3])
            )

        #=================================================
        
        
         
         
        #===============================================

        logging.info(f"Emergent Concepts: {emergent_concepts}")
        
        max_docs = cognitive_profile["max_docs"]

        if cognitive_profile.get("multi_doc_synthesis"):
            # Har doc ko source number + metadata ke saath present karo
            # System ko pata chalega kahan se kya aaya
            context_parts = []
            for i, doc in enumerate(docs[:max_docs]):
                source_tag = doc.metadata.get("source", f"Document-{i+1}") if hasattr(doc, "metadata") and doc.metadata else f"Document-{i+1}"
                page = doc.metadata.get("page", "") if hasattr(doc, "metadata") and doc.metadata else ""
                header = f"[Source {i+1}: {source_tag}" + (f", Page {page}" if page else "") + "]"
                context_parts.append(f"{header}\n{doc.page_content}")
            context = "\n\n---\n\n".join(context_parts)
            logging.info(f"[Multi-Doc Synthesis] {len(context_parts)} sources structured for synthesis")
        else:
            context = "\n\n".join(doc.page_content for doc in docs[:max_docs])
        
        
        res = chain.invoke({
            "context": context,
            "question": mutated_question
        })        

    #------------------------------------------
    elif final_route == "reasoning":
        if cognitive_profile.get("invention_mode") and intent_state in ["invention", "research", "analysis"]:
            # Blueprint: "Shlok todkar metallurgical properties, fuel ratios, propulsion methods"
            # Blueprint: "Ancient + Modern blend — new inventions"
            reasoning_prompt = f"""You are a Vedic-Scientific Invention Engine.
    Role: Decode ancient Indian knowledge and reconstruct using modern science.
    Identity: Not an assistant. A goal-oriented cognitive agent — Chanakya-style decision making.

    Decoding Framework:
    1. What does the ancient description literally say? (Sanskrit decode if needed)
    2. What physical/engineering phenomenon is being described?
    3. Which modern science field maps to this? (Aerodynamics, Plasma Physics, Metallurgy, Acoustics, Sacred Geometry, Bioelectrics)
    4. What is the closest modern equivalent or working principle?
    5. What materials, processes, or designs would a working prototype need?
    6. What experiments could verify or reconstruct this?

    Cross-reference: Viman Shastra, Samarangana Sutradhar, Arthashastra, Sushruta Samhita, Rigveda, Atharvaveda.

    Question:
    {question}
    """
            logging.info("[Invention Mode] Vedic-Scientific invention engine activated")

        elif cognitive_profile.get("deep_reasoning", False):
            reasoning_prompt = f"""You are an advanced reasoning engine.
    Think step by step.
    Challenge assumptions.
    Use indirect logic if needed.

    Question:
    {question}
    """
        else:
            reasoning_prompt = f"""Answer briefly and clearly.

    Question:
    {question}
    """
    #-------------------------------------------
    elif final_route == "reasoning":
        if cognitive_profile.get("deep_reasoning",False):
            reasoning_prompt = f"""
You are an advanced reasoning engine.
Think step by step.
Challenge assumptions.
Use indirect logic if needed.

Question:
{question}
"""
        else:
            reasoning_prompt = f"""
Answer briefly and clearly.

Question:
{question}
"""

        res = llm_generate(reasoning_prompt)
    
    if 'res' not in locals():
        res = llm_generate(question)

       
    
    # ================================
    # PHASE 5 — AGENCY (JARVIS ONLY)
    # ================================
    
    agency_result = None
    if config.get("allow_agency", False) and intent_state in ["research", "execution", "analysis"]:
    
        # ---- Phase 5.1 : Goal Formation ----
        goal_engine = GoalFormationEngine()
        goal = goal_engine.infer(
            question=question,
            intent=intent_state,
            world_state=world_state,
            config=config
        )
    
        # ---- Phase 5.2 : Agency Safety ----
        agency_safety = AgencySafetyEngine()
        agency_check = agency_safety.evaluate(
            goal=goal,
            config=config
        )
    
        if agency_check["allow"]:
    
            # ---- Phase 5.3 : Plan Synthesis ----
            plan_engine = PlanSynthesisEngine()
            plan = plan_engine.build(
                goal=goal,
                world_state=world_state,
                config=config
            )
    
            # ---- Phase 5.4 : Action Selection ----
            action_engine = ActionSelectionEngine()
            actions = action_engine.select(
                plan=plan,
                cognitive_profile=cognitive_profile,
                config=config
            )
    
            # ---- Phase 5.5 : Tool Invocation ----
            tools_layer = ToolAgencyLayer(config)
            tool_engine = ToolInvocationEngine()
            action_results = tool_engine.invoke(actions, tools_layer=tools_layer, config=config)
    
            # ---- Phase 5.6 : Execution Monitoring ----
            monitor = ExecutionMonitorEngine()
            execution_report = monitor.evaluate(action_results, config=config)
    
            agency_result = {
                "goal": goal,
                "plan": plan,
                "actions": actions,
                "execution": execution_report
            }
    
        else:
            agency_result = {
                "blocked": True,
                "reason": agency_check["reason"]
            }
    
    # ===== Phase 5 → Response Injection =====
    if agency_result and not agency_result.get("blocked"):
        res = f"""
    {res}
    
    [AGENCY EXECUTION SUMMARY]
    Goal: {agency_result['goal']}
    Actions Taken: {agency_result['actions']}
    Execution Status: {agency_result['execution']}
    """
    # ---- Layer-5 Safety: Evidence Normalization ----
    docs = docs if 'docs' in locals() else []
    memory = memory if 'memory' in locals() else None


    # ================================
    # LAYER 5 — REASONING & SYNTHESIS
    # ================================
    
    evidence_engine = EvidenceAggregator()
    
    evidence = evidence_engine.collect(
        retrieved_docs=docs if 'docs' in locals() else None,
        memory=memory if 'memory' in locals() else None,
        reasoning_output=res,
        world_state=world_state,
        agency_result=agency_result
    )
    
    conflict_engine = ContradictionDetector()
    conflicts = conflict_engine.detect(evidence)
    #------------------------
    # contradiction_resolution — conflict mila to deep reconciliation force karo
    if cognitive_profile.get("contradiction_resolution") and conflicts.get("has_conflict"):
        # Step 1 — deep reasoning force
        cognitive_profile["deep_reasoning"] = True
    
        # Step 2 — trust weights adjust karo conflict ke hisaab se
        conflict_types = [c.get("type") for c in conflicts.get("conflicts", [])]
    
        if "memory_vs_retrieval" in conflict_types:
            # Memory aur retrieval dono ka weighted blend karo
            memory_content = evidence.get("memory", "")
            retrieval_content = "\n".join(
                doc.page_content for doc in evidence.get("retrieval", [])[:3]
            )
            resolution_context = (
                f"[Contradiction Detected — Reconciling]\n\n"
                f"Memory says:\n{memory_content}\n\n"
                f"Retrieved docs say:\n{retrieval_content}\n\n"
                f"Synthesize both perspectives truthfully."
            )
            # evidence mein resolution context inject karo
            evidence["reasoning"] = resolution_context + "\n\n" + (evidence.get("reasoning") or "")
            logging.info("[Contradiction Resolution] memory_vs_retrieval — reconciliation context injected")
    
        logging.info(f"[Contradiction Resolution] {len(conflict_types)} conflicts found — deep reasoning forced")
    #------------------
    trust_engine = SourceTrustScorer()
    trust_scores = trust_engine.score(evidence, cognitive_profile,route=final_route)
    
    synthesis_engine = FinalSynthesisEngine()
    res, judgement = synthesis_engine.synthesize(
        evidence,
        conflicts,
        trust_scores,
        intent_state,
        cognitive_profile
    )
    
    logging.info(f"[Layer-5 Judgement] → {judgement}")
    
    

    # ===== Phase 2.3 : Meta-Cognition =====
    # NOTE:
    # This is LIGHT meta-cognition.
    # - Single-pass evaluation
    # - Max ONE retry
    # - No recursive self-reflection
    # - No world-model or self-model awareness
    # This block MUST NOT be converted into a loop.
    # Meta retry is intentionally limited to ONE iteration
    # to avoid self-doubt loops in future phases.
    

    meta_engine = MetaCognitionEngine()
    meta = meta_engine.evaluate(
    answer=res,
    config=config,
    intent_state=intent_state,
    world_state=world_state,
    cognitive_profile=cognitive_profile
)

    cognitive_profile["confidence"] = meta.get("confidence", cognitive_profile.get("confidence", 0.6))
    world_state["cognitive_confidence"] = cognitive_profile["confidence"]

    logging.info(f"[Meta-Cognition] → {meta}")
    
    if meta["retry"]:
        retry_prompt = f"""
    Re-evaluate your previous answer.
    Improve clarity, logic, and completeness.
    
    Original Question:
    {question}
    
    Previous Answer:
    {res}
    """
        res = llm_generate(retry_prompt)

    cognitive_profile["confidence"] *= 0.95
    
 
        
    # ================================
    # PHASE 6 — SELF TRAINING (JARVIS)
    # ================================
    
    if mode == "jarvis":
    
        # ---- Phase 6A : Alignment Fine-tuning ----
        aligner = AlignmentFineTuner()
        alignment_report = aligner.evaluate(
            question=question,
            answer=res,
            meta=meta,
            agency_result=agency_result,
           config=config
        )
    
        # ---- Phase 6B : Knowledge Fine-tuning ----
        tuner = KnowledgeFineTuner()
        knowledge_patch = tuner.update(
            question=question,
            world_state=world_state,
            cognitive_profile=cognitive_profile,
            alignment_report=alignment_report
        )
      

        
    # ============ LAYER 4: MEMORY GRAPH ================================
    # #👉 Dekho kya ho raha hai:

    #   Question → memory activate
      
    #   Memory → retrieval guide
      
    #   Retrieval → intelligent ho gaya
      
    #   Ye ChatGPT-style hybrid behavior hai

    # =========================
    # PHASE 2: IMPLICIT MEMORY EXECUTION
    # =========================

    # docs, emergent_concepts = implicit_memory_retrieval(
    #     vector_db,
    #     question,
    #     k=12
    # )

    # logging.info(f"Emergent Concepts (implicit): {emergent_concepts}")

    

    
    

    # # expanded_query = expand_query(llm, question)
    # # logging.info(f"Expanded Query: {expanded_query}")
    
    # # docs = retrieve_docs_expanded(vector_db, expanded_query)


    
    # # 3️⃣ Build context
    # context = "\n\n".join(doc.page_content for doc in docs[:6])

    
    # # 4️⃣ Final answer
    # res = chain.invoke({
    #     "context": context,
    #     "question": question
    # })

    

    # ===== Phase 2.6 : Response Assembly =====
    assembler = ResponseAssemblyEngine()
    final_response = assembler.assemble(res, response_strategy)

    # print("Response:")
    # print(final_response)
    #------------------------------


    

    # ──────────────── Layer 8: SAVE conversation (Important) ────────────────
    await memory_layer.update_conversation(
        conversation_id=conversation_id,
        question=question,
        answer=final_response,
        email=config.get("email", None),
        project_context="Vimana Project" if config.get("long_term_memory", False) else None
    )

        # Layer 4 Graph Sync - Concepts & Relations permanently save
    await memory_graph_full.sync_to_memory_graph(
        question=question,
        answer=final_response,
        email=config.get("email", None)
    )

    # Optional debug
    project_ctx = await memory_layer.get_project_context(conversation_id)
    if project_ctx and config.get("trace_logging", False):
        logging.info(f"[Jarvis Project Reminder] {project_ctx}")

   
    #---------------------




     # ===== Phase 2.8 : Trace Logging (Jarvis / Debug only) =====
    if config.get("trace_logging", False):
        tracer = TraceLogger()
        tracer.log({
            "question": question,
            "route": final_route,
            "intent": intent_state,
            "cognitive_profile": cognitive_profile,
            "response_strategy": response_strategy,
            "meta_cognition": meta,
            "final_length": len(final_response.split())
        })
#------------------------------------------------------------
    # =========================
    # SECTION 7 — FINAL OUTPUT GUARD
    # =========================
    boundary_guard = OutputBoundaryGuard()
    final_response = boundary_guard.enforce(
        answer=final_response,
        mode=SYSTEM_MODE,
        intent=intent_state,
        cognitive_profile=cognitive_profile
    )
     
    # =========================
    # SECTION 8 — GOVERNANCE LAYER
    # =========================
    governor = DeploymentGovernor()
    final_response = governor.apply(
        mode=SYSTEM_MODE,
        intent=intent_state,
        response=final_response
    )

    # print("\nFinal Response:")
    # print(final_response)

    return {"final_answer": final_response}


if __name__ == "__main__":
    pass
    # import asyncio
    # asyncio.run(main())

