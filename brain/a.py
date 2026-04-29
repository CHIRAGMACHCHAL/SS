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