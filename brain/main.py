# main.py
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import uuid
from llm.llm_engine import llm, generate as llm_generate
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
import logging
import spacy
from sentence_transformers import SentenceTransformer
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


# -------- Phase 3.0 : Question Classification --------
class KnowledgeSourceClassifier:
    def classify(self, layer1_bundle: dict) -> str:
        """
        Language-agnostic classification.
        No hardcoded strings — numeric depth + spaCy signals + Layer 1 sub_goals.
        """
        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }

        required_depth   = layer1_bundle.get("required_depth", "normal")
        depth_idx        = DEPTH_INDEX.get(required_depth, 1)
        normalized_query = layer1_bundle.get("normalized_query", "")
        sub_goals        = layer1_bundle.get("sub_goals", [])   # Layer 1 output — complexity signal

        # Layer 1 Phase 1.2 signals — real power
        is_analytical      = layer1_bundle.get("is_analytical", False)
        has_verb           = layer1_bundle.get("has_verb", False)
        has_entity         = layer1_bundle.get("has_entity", False)
        graph_intent_score = layer1_bundle.get("graph_intent_score", 0.0)

        # spaCy — language-agnostic
        doc          = _NLP(normalized_query)
        entity_count = len(doc.ents)
        verb_count   = sum(1 for t in doc if t.pos_ == "VERB")
        noun_count   = sum(1 for t in doc if t.pos_ == "NOUN")

        # sub_goals count → complexity signal from Layer 1
        domains    = layer1_bundle.get("domains", [])
        is_complex = len(sub_goals) >= 3 or len(domains) >= 2   # zyada goals = mixed/complex query


        # REAL POWER: depth deep → conceptual
        if depth_idx >= 3:
            return "conceptual"

        # REAL POWER: is_analytical → causal structure = deep meaning chahiye
        if is_analytical:
            return "conceptual"

        # REAL POWER: Memory Graph strongly activated → conceptual
        if graph_intent_score > 0.6:
            return "conceptual"

        # Shallow depth
        if depth_idx <= 1:
            # REAL POWER: entity present + not analytical → factual
            if has_entity or entity_count >= 2:
                return "factual"
            return "general"

        # Moderate depth
        if is_complex:
            return "mixed"

        # REAL POWER: verb heavy + no entity → procedural
        if has_verb and not has_entity and verb_count > noun_count:
            return "procedural"

        # REAL POWER: entity present → factual
        if has_entity or entity_count >= 2:
            return "factual"

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
        category = classifier.classify(bundle)

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
                config: dict = None, memory_graph=None) -> dict:

        config= config or {}
        
        

        # ════════════════════════════════════
        # PHASE 1.0 — Raw Query Capture & Modality Separation
        # Blueprint: "User ne jo bola exact, bina interpretation"
        # The true bulletproof entry point for all API formats.
        # ════════════════════════════════════
        
        # Phase 1.0 Integration - Blueprint Compliant
        from phase1_0_integration import Phase1_0_Integration
        phase1_int = Phase1_0_Integration()
        
        # Capture user query with all modalities
        phase1_result = phase1_int.capture_user_query(user_query)
        
        # Phase 1.0 output extract karo — Phase 1.1 ke liye
        raw_query      = phase1_result.get("raw_query", "")
        media_payloads = phase1_result.get("media_payloads", {})
        signal_type    = phase1_result.get("signal_type", "text")

        # Safety guard — None ya empty
        if raw_query is None:
            raw_query = ""
            logging.warning("[Phase 1.0] None raw_query — empty string set")

        # Modality detect karo — Llama 3.1 (text) ya Llama 4 (vision)
        # signal_type "image" ya "multimodal" = vision route
        # signal_type "text" = text route
        if signal_type in ("image", "video", "multimodal"):
            state["layer1_modality"] = "vision"   # Llama 4 route
        elif signal_type in ("audio",):
            state["layer1_modality"] = "audio"    # ASR route (future)
        elif signal_type in ("document",):
            state["layer1_modality"] = "document" # Doc extractor route (future)
        else:
            state["layer1_modality"] = "text"     # Llama 3.1 route

        # State mein store karo
        state["layer1_raw_query"]  = raw_query
        state["layer1_media"]      = media_payloads
        state["layer1_signal_type"] = signal_type

        logging.info(
            f"[Phase 1.0] signal_type={signal_type} | "
            f"modality={state['layer1_modality']} | "
            f"raw_query='{raw_query[:60]}'"
        )



        # ════════════════════════════════════
        # PHASE 1.1 — Linguistic Normalization
        # Blueprint: "User ki language broken ho sakti hai, emotional ho sakti
        #             hai, shorthand ho sakti hai. normalize, noise hatao,
        #             grammar perfect karne ki koshish karo"
        #
        # Har tier ke liye same — normalization common zaroori kaam hai.
        # Tier separation nahi — Brain: No subscription awareness.
        # Brain = Code. LLM = Organ (Grammar fix LLM se nahi, Embedder se)
        # ════════════════════════════════════

        # ── MEDIA-ONLY GUARD ─────────────────────────────────────────────────
        # Phase 1.0 se agar sirf bytes/audio/image aaya tha → raw_query = ""
        # Is case mein NLP skip karo, media flag set karo, Phase 1.7 bundle pe jao
        if not raw_query.strip():
            if media_payloads:
                state["layer1_is_media_only"] = True
                logging.info("[Layer1 Ph1.1] MEDIA-ONLY query — NLP skipped, routing to Vision/Audio tools")
            else:
                state["layer1_is_media_only"] = False
                logging.warning("[Layer1 Ph1.1] EMPTY query received — all signals will be zero")

            normalized_query = ""
            detected_lang    = "unknown"
            doc              = _NLP("")
            query_emb        = _EMBEDDER.encode("")
            entities         = []
            noun_chunks      = []

        else:
            state["layer1_is_media_only"] = False

            # ── Step 1: Noise Removal ─────────────────────────────────────────
            # Blueprint: "noise hatao"
            # ftfy — Unicode corruption, broken encoding, mojibake sab fix karta hai
            # Production grade — Wikipedia, CommonCrawl jaise large corpora use karte hain
            cleaned = ftfy.fix_text(raw_query)
            cleaned = re.sub(r'\s+', ' ', cleaned.strip())

            # ── Step 2: Language Detection ────────────────────────────────────
            # 55+ languages — Europe, Asia, Middle East, South Asia sab covered
        
            try:
                detected_lang = lang_detect(cleaned)
            except Exception:
                detected_lang = "en"
        
            # ── Step 3: Grammar Normalization/Deterministic Normalization ────────────────────────────────
            # Blueprint: "grammar perfect karne ki koshish karo"
            # DANGEROUS LLM CALL REMOVED — Blueprint: "Brain = Code, LLM = Organ"
            # LLM hallucinate karke user ka intent badal sakta tha.
            # _EMBEDDER aur _NLP organically broken grammar ko semantic space mein
            # handle karte hain — "plz hlp" aur "please help" ek hi vector point pe map honge.
            normalized_query = cleaned


            # ── Step 4: spaCy parse on PURE exact query ───────────────────────
            # Pehle normalize (noise hata), phir parse — raw par nahi
            doc       = _NLP(normalized_query)
            query_emb = _EMBEDDER.encode(normalized_query)


            # ── Step 5: Linguistic signals (No Billing Limits) ────────────────
            # Blueprint: Base perception billing se restrict nahi hoti
            # Bilkul poora sentence AGI padhega — koi slice nahi
            entities    = [(ent.text, ent.label_) for ent in doc.ents]
            # noun_chunks depends on spaCy pipeline components; fallback to empty on failure.
            try:
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            except Exception:
                noun_chunks = []
            logging.info(
                 f"[Layer1 Ph1.1] query='{normalized_query[:60]}' | "
                 f"lang={detected_lang} | "
                 f"entities={len(entities)} | nouns={len(noun_chunks)}"
            )
        

        # ════════════════════════════════════
        # PHASE 1.2 — Intent Type Detection
        # Blueprint: "ye routing nahi hai, thinking style hai"
        # Blueprint: "Bina Memory Graph ke Intent sirf text hoga, Meaning nahi"
        # Real computed signals — koi string labels nahi
        # ════════════════════════════════════

        # ── spaCy structural signals — language agnostic ──────────────────
        # dep_ aur pos_ universal hain — French, Hindi, Arabic sab mein same
        is_analytical = any(t.dep_ in ("advcl", "ccomp", "expl") for t in doc)
        has_verb      = any(t.pos_ == "VERB" for t in doc)
        has_entity    = len(entities) >= 1

        # Phase 1.2 (Thinking-Type signals)
        # Blueprint: Phase 1.2 intent type detection is thinking-style, not answer routing.
        # These are structural booleans/counts derived from spaCy universal tags (language-agnostic).
        verb_count     = sum(1 for t in doc if t.pos_ == "VERB")
        entity_count   = len(entities)
        noun_count     = len(noun_chunks)

        has_causal_structure   = any(t.dep_ in ["advcl", "mark", "csubj", "relcl"] for t in doc)
        has_process_structure  = verb_count >= 2 and entity_count == 0
        has_factual_structure  = entity_count >= 1 and noun_count >= 1

        # ── Memory Graph semantic activation — Phase 6 ke baad rich hoga ──
        graph_intent_score = 0.0
        if memory_graph is not None:
            try:
                activated = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=3
                )
                if activated:
                    graph_intent_score = activated[0].get("score", 0.0)
                logging.info(f"[Layer1 Ph1.2] graph_intent_score={graph_intent_score:.2f}")
            except Exception as e:
                logging.error(f"[Layer1 Ph1.2] Graph error: {e}")

        logging.info(
            f"[Layer1 Ph1.2] is_analytical={is_analytical} | "
            f"has_verb={has_verb} | has_entity={has_entity} | "
            f"graph_score={graph_intent_score:.2f}"
        )


        # ════════════════════════════════════
        # PHASE 1.3 — Goal Decomposition (HEART of Layer 1)
        # Blueprint: "ChatGPT ek question ko multiple invisible goals mein break karta hai"
        # Blueprint: "Brain: No subscription awareness — sirf config se power milti hai"
        # Blueprint: "Bina Memory Graph ke Intent sirf text hoga, Meaning nahi"
        #
        # Primary: Memory Graph — concepts as goals (semantic, no labels)
        # Fallback: spaCy noun chunks — explicit sub-topics from query structure
        # No hardcoded goal strings | No tier names in brain
        # ════════════════════════════════════

        # max_goals billing.py se config mein inject hota hai — brain reads, never decides
        max_goals = config.get("max_goals", 2)

        sub_goals = []

        # ── Primary: Memory Graph — hidden goals nikaalo ─────────────────
        # Blueprint: "multiple invisible goals" — graph se emerge hote hain
        # Phase 6 training ke baad ye powerful hoga
        if memory_graph is not None:
            try:
                activated = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=max_goals
                )
                sub_goals = [
                    c["concept"] for c in activated
                    if c.get("score", 0) > 0.3
                ]
            except Exception as e:
                logging.warning(f"[Layer1 Ph1.3] Memory graph error: {e}")

        # ── Fallback: spaCy noun chunks ───────────────────────────────────
        # Har noun chunk query ka ek sub-topic hai — language agnostic
        # Ye actual query se nikle hain — hardcoded nahi
        if len(sub_goals) < max_goals and noun_chunks:
            for chunk in noun_chunks:
                if chunk not in sub_goals:
                    sub_goals.append(chunk)
                if len(sub_goals) >= max_goals:
                    break

        # ── Final cap ────────────────────────────────────────────────────
        sub_goals = sub_goals[:max_goals]

        logging.info(
            f"[Layer1 Ph1.3] sub_goals={sub_goals} | max_goals={max_goals}"
        )
        # ════════════════════════════════════
        # PHASE 1.4 — Query Expansion
        # Blueprint: "ChatGPT internally expanded semantic query banata hai"
        # Blueprint: "Brain: No subscription awareness — config se power milti hai"
        #
        # Primary: Memory Graph semantic neighbors — concept-driven expansion
        # Secondary: spaCy entities — linguistic expansion
        # No hardcoded templates | No tier names in brain | Language agnostic
        # ════════════════════════════════════

        # max_expansion billing.py se config mein inject hota hai
        max_expansion = config.get("max_query_expansion", 1)

        expanded_queries = [normalized_query]  # original normalized query
        temp_expanded    = []

        # ── Primary: Memory Graph neighbors — semantic expansion ──────────
        # Blueprint: "ChatGPT internally expanded semantic query banata hai"
        # Graph trained hoga Phase 6 ke baad — tab rich expansion milegi
        # Abhi graph khali hai to spaCy fallback chalega — ye sahi behaviour hai
        if memory_graph is not None:
            try:
                neighbors = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=max_expansion
                )
                for n in neighbors:
                    concept = n.get("concept", "")
                    score   = n.get("score", 0)
                    if concept and score > 0.3 and concept.lower() not in normalized_query.lower():
                        temp_expanded.append(f"{normalized_query} {concept}")
            except Exception as e:
                logging.warning(f"[Layer1 Ph1.4] Graph expansion error: {e}")

        # ── Secondary: spaCy entities — linguistic expansion ─────────────
        # Entities = query ke important concepts — language agnostic
        if entities and len(temp_expanded) < max_expansion:
            entity_texts   = [e[0] for e in entities[:3]]
            entity_variant = f"{normalized_query} {' '.join(entity_texts)}"
            if entity_variant.strip() != normalized_query.strip():
                temp_expanded.append(entity_variant)

        # ── Apply cap aur merge ───────────────────────────────────────────
        for v in temp_expanded:
            if v not in expanded_queries and len(expanded_queries) < max_expansion:
                expanded_queries.append(v)

        logging.info(
            f"[Layer1 Ph1.4] expanded={len(expanded_queries)} queries | "
            f"max={max_expansion}"
        )

        # ════════════════════════════════════
        # PHASE 1.5 — Reasoning Depth Estimation
        # Blueprint: "Short answer chalega ya heavy multi-layer reasoning?"
        # Blueprint: "Brain: No subscription awareness — config se power milti hai"
        #
        # Pure objective signals — koi intent labels nahi, koi tier names nahi
        # Memory Graph relevance + spaCy complexity + sub_goals count
        # ════════════════════════════════════

        # ── Objective signals ─────────────────────────────────────────────
        n_entities  = len(entities)
        goal_count  = len(sub_goals)

        # Memory Graph relevance — Phase 6 ke baad strong signal dega
        graph_relevance = 0.0
        if memory_graph is not None:
            try:
                graph_relevance = memory_graph.estimate_relevance(normalized_query)
            except Exception:
                pass

        # ── Depth decision — pure signals, koi tier naam nahi ────────────
        # Sirf objective measures: graph strength, goals, entities, sentences
        if graph_relevance > 0.8 or goal_count >= 5:
            required_depth = "ultra_deep"
        elif graph_relevance > 0.7 or goal_count >= 4:
            required_depth = "very_deep"
        elif goal_count >= 3 or n_entities >= 3:
            required_depth = "deep"
        elif goal_count >= 2 or n_entities >= 2:
            required_depth = "moderate"
        elif goal_count >= 1 or n_entities >= 1:
            required_depth = "normal"
        else:
            required_depth = "shallow"

        # ── Tier cap — billing.py se inject hota hai, brain decide nahi karta
        max_allowed_depth = config.get("max_depth", "shallow")
        depth_levels = ["shallow", "normal", "moderate", "deep", "very_deep", "ultra_deep"]
        max_idx           = depth_levels.index(max_allowed_depth)
        current_idx       = depth_levels.index(required_depth)

        if current_idx > max_idx:
            required_depth = max_allowed_depth
            current_idx    = max_idx

        logging.info(
            f"[Layer1 Ph1.5] depth={required_depth} | "
            f"ents={n_entities} | "
            f"goals={goal_count} | graph={graph_relevance:.2f}"
        )
        # ════════════════════════════════════
        # PHASE 1.6 — Knowledge Domain Mapping
        # Blueprint: "Concepts likhe nahi jaate — nikal ke aate hain"
        # Blueprint: "Brain: No subscription awareness — config se power milti hai"
        #
        # Primary: Memory Graph — domains emerge hote hain Phase 6 training ke baad
        # Fallback: spaCy entities — query se hi domain signal aata hai
        # allow_ancient_tech billing.py se inject hota hai — brain decide nahi karta
        # ════════════════════════════════════

        domains = []

        # ── Primary: Memory Graph se domain emergence ─────────────────────
        # Blueprint: "Bina Memory Graph ke Intent sirf text hoga, Meaning nahi"
        # Phase 6 training ke baad graph rich hoga — domains khud emerge honge
        if memory_graph is not None:
            try:
                activated = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=5
                )
                domains = [
                    c["concept"] for c in activated
                    if c.get("score", 0) > 0.35
                ]
            except Exception as e:
                logging.warning(f"[Layer1 Ph1.6] Graph domain error: {e}")

        # ── Fallback: spaCy entities ──────────────────────────────────────
        # Query mein jo entities hain — wahi domain hint deti hain
        # Language agnostic — kisi bhi language mein kaam karega
        if not domains and entities:
            domains = [ent[0] for ent in entities[:3]]

        if not domains:
            domains = ["general"]

        logging.info(f"[Layer1 Ph1.6] domains={domains}")

        # ════════════════════════════════════
        # PHASE 1.7 — Intent Bundle
        # Blueprint: "Layer 1 ka output ek object hota hai
        #              jo SAARI layers use karti hain"
        # ════════════════════════════════════
        intent_bundle = {
            "raw_query":         raw_query,
            "normalized_query":  normalized_query,
            "query_embedding":   query_emb.tolist(),
            # "intent_type":       intent_type,
            # "thinking_type":     thinking_type,
            "sub_goals":         sub_goals,
            "expanded_queries":  expanded_queries,
            "required_depth":    required_depth,
            "domains":           domains,
            # "reasoning_plan":    sub_goals[:3],
            # spaCy signals — Layer 2, 3 ke liye
            "entities":          entities,
            "noun_chunks":       noun_chunks,
            # Phase 1.2 — thinking-style signals (computed; no hardcoded user keywords)
            "verb_count":        verb_count,
            "noun_count":        noun_count,
            "has_causal_structure":  has_causal_structure,
            "has_process_structure": has_process_structure,
            "has_factual_structure": has_factual_structure,
            # Phase 1.7 — reasoning plan (not answer): derived from hidden goals
            "reasoning_plan":   sub_goals[:3],
            # billing config value — Layer 2 branching ke liye
            "max_goals":         max_goals,
            "is_analytical":       is_analytical,
            "has_verb":            has_verb,
            "has_entity":          has_entity,
            "graph_intent_score":  graph_intent_score,
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

    Drop   = score > 0.85 — strongly in memory, repeat waste hogi
    Merge  = duplicate strings — seen set se deduplicate
    Deeper = score low + required_depth deep — naya topic, rakhna zaroori
    Koi string label nahi, koi hardcoded English nahi.
    """

    def prune(
        self,
        queries: list,
        memory_graph,
        layer1_bundle: dict
    ) -> list:
        """
        queries       : Phase 2.4 ka output — plain string list
        memory_graph  : Layer 4 hook
        layer1_bundle : required_depth ke liye — billing config se aaya

        Returns: pruned list — drop/merge/deeper decisions applied
        """
        # memory_graph None ho to pruning skip — safe fallback
        if memory_graph is None or not queries:
            return queries


        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }
        required_depth     = layer1_bundle.get("required_depth", "shallow")
        depth_idx          = DEPTH_INDEX.get(required_depth, 0)
        is_deep            = depth_idx >= 3

        # Layer 1 Phase 1.2 signals — real power
        is_analytical      = layer1_bundle.get("is_analytical", False)
        graph_intent_score = layer1_bundle.get("graph_intent_score", 0.0)

        pruned = []
        seen   = set()

        for q in queries:
            # Merge — duplicate deduplicate
            key = q.lower().strip()
            if key in seen:
                continue
            seen.add(key)

            try:
                score = memory_graph.estimate_relevance(q)
            except Exception as e:
                logging.warning(f"[Ph2.5] memory error: {e}")
                pruned.append(q)
                continue

            # Drop — strongly in memory, repeat waste hogi
            # Blueprint: "kya ye knowledge already store hai? drop karo"
            if score > 0.85 and not is_analytical:
                continue

            # Deeper — Blueprint: "score low + required_depth deep = naya topic, rakhna zaroori"
            if is_deep and score < 0.3:
                pruned.append(q)
                continue

            if graph_intent_score > 0.6 and score > 0.3:
                pruned.append(q)
                continue

            # Normal — drop nahi, deeper nahi — as is rakhna hai
            pruned.append(q)

        # Blueprint: fallback — kuch to dena hai Layer 3 ko
        return pruned if pruned else queries

#================================================
#==========LAYER 2 : ADAPTIVE QUERY EXPANSION(DYNAMICS)==========
#================================================
# .......Phase 2.1 : INTENT-wise QUERY BRANCHING.........
class IntentQueryBrancher:
    """
    Blueprint: "har intent ke liye alag query path banana — yhi se system smart lgta hai"
    Blueprint: "decision-based, static nahi"
    Blueprint: "Concepts nikal ke aate hain"

    Layer 1 ne sub_goals (Memory Graph), entities (spaCy), noun_chunks nikale —
    inhe base_query ke saath combine karke genuinely alag query paths banao.
    Koi intent_state label nahi. Koi hardcoded English string nahi.
    Language agnostic — embedding aur spaCy dono language se upar hain.
    """

    def branch(self, base_query: str, layer1_bundle: dict, cognitive_profile: dict = None) -> list:
        """
        base_query    : Layer 1 ka normalized_query
        layer1_bundle : Layer 1 ka poora output
        Returns: list of actual query strings — genuinely alag angles
        Ye strings directly Layer 3 mein search karengi
        """
        sub_goals      = layer1_bundle.get("sub_goals", [])
        entities       = [e[0] for e in layer1_bundle.get("entities", [])]
        noun_chunks    = layer1_bundle.get("noun_chunks", [])
        required_depth = layer1_bundle.get("required_depth", "normal")
        branches = []
        seen = set()
    
        # Layer 1 Phase 1.2 signals — real power
        is_analytical      = layer1_bundle.get("is_analytical", False)
        has_verb           = layer1_bundle.get("has_verb", False)
        has_entity         = layer1_bundle.get("has_entity", False)
        graph_intent_score = layer1_bundle.get("graph_intent_score", 0.0)
    
        # Numeric depth — string match nahi
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
    
        branches = []
        seen = set()
    
        def add(q: str):
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                branches.append(q)
    
        # Branch 1 — base query hamesha
        add(base_query)
    
        # Branch 2 — sub_goals: Memory Graph ke concepts — real semantic angles
        for goal in sub_goals:
            if goal.lower() not in base_query.lower():
                add(f"{base_query} {goal}")
    
        # Branch 3 — entities — complexity ke hisaab se
        if entities:
            add(f"{base_query} {' '.join(entities[:max_branches])}")
    
        # Branch 4 — noun_chunks — complexity ke hisaab se
        for chunk in noun_chunks[:max_branches]:
            if chunk.lower() not in base_query.lower():
                add(f"{chunk} {base_query}")
    
        # Branch 5 — depth numeric >= 3 — extra entity angle
        # REAL POWER: deep query mein zyada angles explore honge
        if depth_idx >= 3:
            for ent in entities[:2]:
                if ent.lower() not in base_query.lower():
                    add(f"{ent} {base_query}")
    
        # Branch 6 — Layer 1 Phase 1.2 REAL POWER
        # is_analytical=True → causal/conceptual angle add karo
        # REAL POWER: analytical query ke liye ek extra conceptual branch
        if is_analytical and graph_intent_score > 0.3:
            for goal in sub_goals[:2]:
                add(f"{goal} {base_query}")
    
        # Branch 7 — has_verb=True, no entity → procedural angle
        # REAL POWER: process-oriented query ke liye verb-first branch
        if has_verb and not has_entity and noun_chunks:
            add(f"{noun_chunks[0]} {base_query}" if noun_chunks else base_query)
    
        return branches
#.............. Phase 2.2 : QUERY GRANULARITY DECISION ..........
class QueryGranularityDecider:
    """
    Blueprint: "HAR INTENT KE LIYE DECIDE HOTA HAI —
                Narrow: exact facts, Broad: Landscape, Abstract: Philosophy"
    Narrow/Broad/Abstract — measure ke 3 ends hain, string labels nahi.
    scope_score 0.0–1.0 — pure number — Phase 2.3 + 2.4 ye directly use karenge.
    """

    def decide(self, branches: list, layer1_bundle: dict, memory_graph) -> list:
        DEPTH_INDEX = {
            "shallow": 0, "normal": 1, "moderate": 2,
            "deep": 3, "very_deep": 4, "ultra_deep": 5
        }

        required_depth = layer1_bundle.get("required_depth", "shallow")
        depth_idx      = DEPTH_INDEX.get(required_depth, 0)
        depth_score    = depth_idx / (len(DEPTH_INDEX) - 1)  # 0.0–1.0 normalize
    
        # Layer 1 Phase 1.2 signals — real power
        is_analytical      = layer1_bundle.get("is_analytical", False)
        has_entity         = layer1_bundle.get("has_entity", False)
        has_verb           = layer1_bundle.get("has_verb", False)
        graph_intent_score = layer1_bundle.get("graph_intent_score", 0.0)
    
        # Intent-based scope bias — REAL POWER
        # is_analytical → abstract scope chahiye (philosophy, theory, causal)
        # has_entity only → narrow scope (exact facts, definitions)
        # has_verb only → broad scope (process, landscape)
        if is_analytical:
            intent_bias = 0.3   # abstract end ki taraf push karo
        elif has_entity and not is_analytical:
            intent_bias = -0.2  # narrow end ki taraf push karo
        elif has_verb and not has_entity:
            intent_bias = 0.1   # broad middle mein raho
        else:
            intent_bias = 0.0
    
        # graph_intent_score — Memory Graph relevant hai to abstract scope zyada useful
        graph_bias = graph_intent_score * 0.2
    
        results = []
        for branch in branches:
            mem_score = 0.0
            if memory_graph is not None:
                try:
                    mem_score = memory_graph.estimate_relevance(branch)
                except Exception as e:
                    logging.warning(f"[Ph2.2] memory error: {e}")
    
            # depth 50% + memory 30% + intent_bias 20% = scope_score
            # REAL POWER: intent type scope decide karta hai, sirf depth nahi
            scope_score = round(
                (depth_score * 0.5) + (mem_score * 0.3) + intent_bias + graph_bias,
                4
            )
            # 0.0–1.0 clamp
            scope_score = max(0.0, min(1.0, scope_score))
            results.append({"branch": branch, "scope_score": scope_score})
    
        logging.info(f"[Ph2.2] scope_scores computed for {len(results)} branches | intent_bias={intent_bias}")
        return results

#.................. Phase 2.3 : DYNAMIC QUERY SHAPE GENERATOR ..........
class QueryShapeGenerator:
    """
    Blueprint: "yahaan actually query forms bante hai —
                Declaration, Exploratory, Hypothetical, Comparative, Causal.
                ye phase random nhi hota, decision-based hota hai."
    Decision = Phase 2.2 ka scope_score.
    Forms = Memory Graph ke close concepts se emerge karte hain.
    Koi hardcoded English nahi — language agnostic.
    """
    def generate(
        self,
        branch_item: dict,
        query_embedding: list,
        memory_graph,
        cognitive_profile: dict = None,
        layer1_bundle: dict = None       # Layer 1 signals ke liye
    ) -> str:
        branch      = branch_item["branch"]
        scope_score = branch_item["scope_score"]

        # Billing gate — FREE/PAID ke liye emergent concepts off
        if not (cognitive_profile or {}).get("use_emergent_concepts", False):
            return branch

        # Narrow query — already specific, enrichment noise banega
        if scope_score < 0.2 or memory_graph is None or not query_embedding:
            return branch

        # Layer 1 Phase 1.2 signals — real power
        bundle             = layer1_bundle or {}
        is_analytical      = bundle.get("is_analytical", False)
        has_verb           = bundle.get("has_verb", False)
        has_entity         = bundle.get("has_entity", False)
        graph_intent_score = bundle.get("graph_intent_score", 0.0)

        try:
            # scope_score se top_k decide
            top_k = max(1, round(scope_score * 3))
            similar = memory_graph.get_similar_concepts(query_embedding, top_k=top_k)

            # Close concepts — score > 0.5
            close_concepts = [
                c["concept"] for c in similar
                if c.get("score", 0) > 0.5
                and c["concept"].lower() not in branch.lower()
            ]

            if not close_concepts:
                return branch

            # REAL POWER — Query shape Layer 1 signals se decide hoti hai
            # Blueprint: "Declaration, Exploratory, Hypothetical, Comparative, Causal"

            # Causal shape — is_analytical=True matlab causal structure hai
            # REAL POWER: analytical query → concepts ko causal angle se connect karo
            if is_analytical:
                return f"{branch} {close_concepts[0]}"

            # Comparative shape — Memory Graph strongly activated
            # REAL POWER: graph relevant → related concepts compare karo
            if graph_intent_score > 0.6 and len(close_concepts) >= 2:
                return f"{branch} {close_concepts[0]} {close_concepts[1]}"            

            # Exploratory shape — high scope, wide angle
            # REAL POWER: abstract query → multiple concepts se explore karo
            if scope_score > 0.7:
                return f"{branch} {' '.join(close_concepts[:3])}"

            # Declaration shape — entity present, narrow scope
            # REAL POWER: factual query → entity pe focused single concept
            if has_entity and scope_score < 0.5:
                return f"{branch} {close_concepts[0]}"

            # Procedural shape — verb heavy
            # REAL POWER: process query → action concept add karo
            if has_verb and not has_entity:
                return f"{branch} {close_concepts[0]}"

            # Default — concept add karo
            return f"{branch} {' '.join(close_concepts)}"

        except Exception as e:
            logging.warning(f"[Ph2.3] memory error: {e}")

        return branch
 
#...............Phase 2.4 : ABSTRACTION LEVEL MODULATOR ..........
class AbstractionModulator:
    """
    Blueprint: "Same intent ko multiple abstract levels pr query krta hai —
                Concrete (facts), Conceptual (models), Meta (ethics, philosophy)"
    Phase 2.3 ne close concepts liye (concrete form).
    Phase 2.4 door ke concepts leta hai — graph distance = abstraction level.
    Blueprint ka "Meta — philosophy" graph mein door ke nodes se aata hai.
    Koi hardcoded English nahi.
    """

    def adjust(
        self,
        query: str,
        scope_score: float,
        query_embedding: list,
        memory_graph,
        cognitive_profile: dict = None,
        layer1_bundle: dict = None        # Layer 1 signals
    ) -> str:

        # Billing gate
        if not (cognitive_profile or {}).get("use_emergent_concepts", False):
            return query

        if scope_score < 0.5 or memory_graph is None or not query_embedding:
            return query

        # Layer 1 Phase 1.2 signals
        bundle        = layer1_bundle or {}
        is_analytical = bundle.get("is_analytical", False)
        has_entity    = bundle.get("has_entity", False)

        # has_entity + low analytical = Concrete level sufficient
        # REAL POWER: factual query ko abstract mat karo — noise banega
        if has_entity and not is_analytical and scope_score < 0.7:
            return query

        try:
            similar = memory_graph.get_similar_concepts(query_embedding, top_k=8)

            # ── Conceptual level — scope 0.65–0.8 ──────────────────────────
            # Medium distance concepts — models, theories angle
            # REAL POWER: moderate abstract query ko theoretical angle milta hai
            if 0.65 <= scope_score <= 0.8:
                conceptual = [
                    c["concept"] for c in similar
                    if 0.3 <= c.get("score", 0) <= 0.5
                    and c["concept"].lower() not in query.lower()
                ]
                if conceptual:
                    return f"{query} {conceptual[0]}"

            # ── Meta level — scope > 0.8 ya is_analytical ──────────────────
            # Door ke concepts — ethics, philosophy angle
            # REAL POWER: deep analytical query ko meta/philosophical angle milta hai
            if scope_score > 0.8 or is_analytical:
                meta = [
                    c["concept"] for c in similar
                    if 0.15 < c.get("score", 0) < 0.3
                    and c["concept"].lower() not in query.lower()
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
        Returns: budget ke andar top-priority queries — sorted
        """
        budget = cognitive_profile.get("max_docs", 6)

        # Layer 1 Phase 1.2 signals — real power
        bundle             = layer1_bundle or {}
        is_analytical      = bundle.get("is_analytical", False)
        graph_intent_score = bundle.get("graph_intent_score", 0.0)
        deep_reasoning     = cognitive_profile.get("deep_reasoning", False)

        def priority_score(q: str) -> float:
            base = 0.5
            if memory_graph is not None:
                try:
                    mem_score = memory_graph.estimate_relevance(q)
                    # Invert — low memory = naya = high priority
                    # Blueprint: "Kaun Pehle?" = unexplored pehle
                    base = 1.0 - mem_score
                except Exception as e:
                    logging.warning(f"[Ph2.6] memory error: {e}")

            # REAL POWER: analytical query ko extra priority
            # Blueprint: "intent importance" factor
            if is_analytical:
                base += 0.2

            # REAL POWER: graph activated queries pehle
            if graph_intent_score > 0.6:
                base += 0.15

            # REAL POWER: deep reasoning — unexplored topics zyada priority
            if deep_reasoning:
                base += 0.1

            return min(base, 1.0)

        prioritized = sorted(queries, key=priority_score, reverse=True)
        logging.info(f"[Ph2.6] budget={budget}, total={len(queries)}, selected={min(budget, len(queries))}")
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

    def run(self, question: str,layer1_bundle: dict, cognitive_profile: dict, memory_graph ) -> list:
        brancher          = IntentQueryBrancher()
        granularity_decider = QueryGranularityDecider()
        shape_gen         = QueryShapeGenerator()
        abstraction       = AbstractionModulator()
        allocator         = QueryBudgetAllocator()

        # Phase 2.1 — branches
        branches = brancher.branch(
            layer1_bundle.get("normalized_query", question),
            layer1_bundle,
            cognitive_profile
        )

        # Phase 2.2 — scope scores
        query_scope_items = granularity_decider.decide(
            branches, layer1_bundle, memory_graph
        )

        # Phase 2.3 + 2.4 — shape + abstraction
        query_emb = layer1_bundle.get("query_embedding")
        queries = []
        for item in query_scope_items:
            q = shape_gen.generate(item, query_emb, memory_graph, cognitive_profile, layer1_bundle)
            q = abstraction.adjust(q, item["scope_score"], query_emb, memory_graph, cognitive_profile, layer1_bundle)
            queries.append(q)

        # Phase 2.5 — prune
        queries = MemoryAwareQueryPruner().prune(queries, memory_graph, layer1_bundle)

        # Phase 2.6 — priority + budget
        queries = allocator.allocate(queries, cognitive_profile, memory_graph, layer1_bundle)

        # Ph 2.8 Trace — billing flag se control
        if cognitive_profile.get("trace_logging", False):
            logging.info(f"[TRACE Ph2.1-branches] {branches}")
            logging.info(f"[TRACE Ph2.2-scope_items] count={len(query_scope_items)}")
            logging.info(f"[TRACE Ph2.3-shape] first_query={queries[0] if queries else 'empty'}")
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
    # mode = "jarvis" if tier == "jarvis" else "public"


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
