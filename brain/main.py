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
        words = text.lower().split()
        return [w for w in words if len(w) > 6]

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

def implicit_memory_retrieval(vector_db, question, k=12):
    """
    Phase-2 implicit memory:
    - No hardcoded concepts
    - Embedding similarity drives memory
    - Concepts emerge from retrieved chunks
    """

    # Step 1: Raw semantic retrieval
    docs = vector_db.similarity_search(question, k=k)

    # Step 2: Extract emergent keywords (lightweight signal)
    concept_counter = {}

    for doc in docs:
        words = doc.page_content.lower().split()
        for w in words:
            if len(w) > 5:   # noise filter
                concept_counter[w] = concept_counter.get(w, 0) + 1

    # Step 3: Top emergent "concept hints"
    emergent_concepts = sorted(
        concept_counter,
        key=concept_counter.get,
        reverse=True
    )[:8]

    return docs, emergent_concepts


# =========================
# PHASE 2.1: COGNITIVE ROUTER
# =========================


class CognitiveRouter:
    def route(self, question: str) -> str:
        q = question.lower()

        if self.is_memory_query(q):
            return "memory"

        if self.is_fact_query(q):
            return "retrieval"

        if self.is_reasoning_query(q):
            return "reasoning"

        return "direct"
    
    def route_with_context(
            self,
            *,
            question: str,
            intent,
            domains,
            required_depth
        ) -> str:
            """
            FULL cognitive routing (heavy, non-lite)
                    """
    
            # 1️⃣ Intent based routing
            if intent in {"reasoning", "planning", "comparison"}:
                primary_route = "reasoning_engine"
    
            elif intent in {"factual", "definition", "lookup"}:
                primary_route = "knowledge_retrieval"
    
            elif intent in {"creative", "story", "idea"}:
                primary_route = "creative_engine"
    
            else:
                primary_route = "hybrid_engine"
    
            # 2️⃣ Domain override
            if domains:
                if "science" in domains or "tech" in domains:
                    primary_route = "knowledge_retrieval"
    
                elif "philosophy" in domains:
                    primary_route = "reasoning_engine"
    
            # 3️⃣ Depth modulation
            if required_depth == "deep":
                if primary_route == "knowledge_retrieval":
                    primary_route = "hybrid_engine"
    
            # 4️⃣ Safety fallback (old router)
            if not primary_route:
                primary_route = self.route(question)
    
            return primary_route

    def is_memory_query(self, q):
        return any(x in q for x in [
            "yaad", "pehle", "tumne kaha", "memory", "earlier"
        ])

    def is_fact_query(self, q):
        return any(x in q for x in [
            "what is", "who", "when", "define", "list"
        ])

    def is_reasoning_query(self, q):
        return any(x in q for x in [
            "why", "how", "explain", "kaise", "kyu"
        ])


def memory_lookup(vector_db, question, k=6):
    docs = vector_db.similarity_search(question, k=k)
    if not docs:
        return None
    return "\n".join(doc.page_content for doc in docs[:3])


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
            if intent_value == "research":
                strategy.update({
                    "style": "deep-analytical",
                    "structure": "vedic-scientific-report",
                    "verbosity": "maximum",
                    "multi_perspective": True,
                    "cross_domain": True
                })
            elif intent_value == "execution":
                strategy.update({
                    "style": "instructional",
                    "structure": "detailed-steps",
                    "verbosity": "maximum",
                    "include_verification": True
                })
            elif intent_value == "conversation":
                strategy.update({
                    "style": "deep-casual",
                    "structure": "free",
                    "verbosity": "high"
                })
            elif intent_value == "planning":
                strategy.update({
                    "style": "strategic-vedic",
                    "structure": "multi-layer-plan",
                    "verbosity": "maximum",
                    "cross_domain": True
                })
            elif intent_value == "analysis":
                strategy.update({
                    "style": "critical-vedic",
                    "structure": "comparative-deep",
                    "verbosity": "maximum",
                    "multi_perspective": True
                })
            elif intent_value == "invention":
                strategy.update({
                    "style": "creative-technical",
                    "structure": "innovation-report",
                    "verbosity": "maximum",
                    "cross_domain": True,
                    "include_verification": True
                })

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

        # assumption_checking active hai to hidden assumptions mutated query mein daal do
        if cognitive_profile.get("assumption_checking") and world_state.get("hidden_assumptions"):
                assumption_context = " | ".join(world_state["hidden_assumptions"][:3])
                mutated_question = mutated_question + f" [Challenge these assumptions: {assumption_context}]"
                logging.info(f"[Assumption Checking] Injected into query")        

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
    def decide(self, route: str, config: dict, world_state: dict, intent: dict | None = None, required_depth: str = "normal"):
        """
        Returns cognitive profile based on billing config + route + world + intent
        Brain ka kaam: world/intent/depth se adjust karna
        Billing ka kaam: base power level inject karna
        """

        cognitive_load_level = config.get("cognitive_load_level", "minimal")

        # ===== MINIMAL (free) =====
        if cognitive_load_level == "minimal":
            profile = {
                "use_chain": True,
                "deep_reasoning": False,
                "use_emergent_concepts": False,
                "max_docs": 4,
                "query_complexity": "low",
                "parallel_thinking": False,
                "assumption_checking": False
            }

        # ===== STANDARD (paid) =====
        elif cognitive_load_level == "standard":
            profile = {
                "use_chain": True,
                "deep_reasoning": False,
                "use_emergent_concepts": True,
                "max_docs": 8,
                "query_complexity": "normal",
                "parallel_thinking": False,
                "assumption_checking": False
            }

        # ===== ADVANCED (ultra_paid) =====
        elif cognitive_load_level == "advanced":
            profile = {
                "use_chain": True,
                "deep_reasoning": True,
                "use_emergent_concepts": True,
                "max_docs": 15,
                "query_complexity": "high",
                "parallel_thinking": False,
                "assumption_checking": True
            }

        # ===== PROFESSIONAL (business_small) — team workload =====
        elif cognitive_load_level == "professional":
            profile = {
                "use_chain": True,
                "deep_reasoning": True,
                "use_emergent_concepts": True,
                "max_docs": 25,
                "query_complexity": "high",
                "parallel_thinking": True,
                "assumption_checking": True,
                "multi_doc_synthesis": True
            }

        # ===== EXPERT (enterprise) — org level heavy load =====
        elif cognitive_load_level == "expert":
            profile = {
                "use_chain": True,
                "deep_reasoning": True,
                "use_emergent_concepts": True,
                "max_docs": 40,
                "query_complexity": "very_high",
                "parallel_thinking": True,
                "assumption_checking": True,
                "multi_doc_synthesis": True,
                "contradiction_resolution": True
            }

        # ===== MAXIMUM (jarvis) — full cognitive power =====
        elif cognitive_load_level == "maximum":
            profile = {
                "use_chain": True,
                "deep_reasoning": True,
                "use_emergent_concepts": True,
                "max_docs": 999,
                "query_complexity": "maximum",
                "parallel_thinking": True,
                "assumption_checking": True,
                "multi_doc_synthesis": True,
                "contradiction_resolution": True,
                "vedic_cross_reference": True,
                "ancient_modern_blend": True,
                "invention_mode": False  # intent se trigger hoga
            }

        else:
            # Fallback — safe default
            profile = {
                "use_chain": True,
                "deep_reasoning": False,
                "use_emergent_concepts": False,
                "max_docs": 4,
                "query_complexity": "low",
                "parallel_thinking": False,
                "assumption_checking": False
            }

        # ================================================================
        # BRAIN KA KAAM — Route/World/Intent/Depth aware adjustments
        # Yeh sab tier se independent hain — pure cognitive logic
        # ================================================================

        # ---- Route-aware ----
        if route == "reasoning":
            profile["deep_reasoning"] = True
            profile["use_chain"] = True

        elif route == "retrieval":
            profile["use_emergent_concepts"] = True
            profile["max_docs"] = max(profile["max_docs"], 6)

        elif route == "memory":
            profile["use_chain"] = False  # memory route mein chain avoid karo

        # ---- World-aware adjustments (brain ka kaam — sahi jagah) ----
        if world_state["domain"] == "political":
            profile["deep_reasoning"] = True
            profile["use_emergent_concepts"] = True
            profile["assumption_checking"] = True

        if world_state.get("human_factor"):
            profile["use_chain"] = False  # avoid cold logic

        if world_state["ethical_weight"] == "medium":
            profile["deep_reasoning"] = True

        # ---- Intent-aware tuning ----
        if intent:
            if isinstance(intent, dict):
                urgency = intent.get("urgency", "normal")
            else:
                urgency = "normal"

            if urgency == "high":
                profile["deep_reasoning"] = True

        # ---- Depth-aware ----
        if required_depth == "deep":
            profile["deep_reasoning"] = True
            profile["max_docs"] = min(profile["max_docs"] + 2, 999)

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

        if meta.get("confidence") == "low":
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
        LIGHT meta-cognition — blueprint strict.
        
        Kaam:
          1. Answer observe karo — judge karo (observer, participant nahi)
          2. Confidence estimate karo — proxy signals se
          3. Retry decide karo — billing config se, tier agnostic
        
        Rules:
          - NO LLM call — brain ka code hai yeh
          - Single-pass evaluation only
          - MAX ONE retry — call site enforce karta hai
          - No loops, no recursion
          - No world-model / self-model awareness (abhi light phase hai)
        """

        # ====================================================
        # STEP 1: ANSWER KE PROXY SIGNALS — observer mode
        # Blueprint: "lightweight signal use karta hai"
        # ====================================================

        length = len(answer.split())

        # Structure hai? — organized thinking ka sign
        has_structure = any(
            m in answer
            for m in ["1.", "2.", "3.", "- ", "• ", "\n\n", "###", "**"]
        )

        # Reasoning markers? — causal thinking ka sign
        has_reasoning = any(
            w in answer.lower()
            for w in [
                "because", "therefore", "hence", "reason", "evidence",
                "suggests", "indicates", "isliye", "kyunki", "iska matlab",
                "consequently", "thus", "as a result"
            ]
        )

        # Incomplete answer? — strong negative signal
        is_incomplete = answer.strip().endswith(
            ("...", "etc", "and so on", "aadi", "etc.")
        )

        # Cognitive profile se — deep reasoning use hua ya nahi
        # Agar deep reasoning use hua aur answer chhota hai — suspicious
        deep_used = (cognitive_profile or {}).get("deep_reasoning", False)

        # ====================================================
        # STEP 2: SCORE CALCULATE KARO — depth proxy
        # Blueprint: f(length, structure, reasoning_steps...)
        # ====================================================

        score = 0

        # Length — depth ka proxy
        if length > 150:    score += 3
        elif length > 80:   score += 2
        elif length > 30:   score += 1
        # < 30 words = likely incomplete, score stays 0

        # Structure — organized answer
        if has_structure:   score += 1

        # Reasoning — causal thinking present
        if has_reasoning:   score += 2

        # Deep reasoning use hua — cognitive investment proof
        if deep_used:       score += 1

        # Incomplete — strong negative
        if is_incomplete:   score -= 3

        # ====================================================
        # STEP 3: CONTEXT-AWARE STANDARD TIGHTEN KARO
        # Billing se alag — yeh answer ki quality standard hai
        # ====================================================

        # World context se — ethical domain mein zyada careful
        ethical_weight = (world_state or {}).get("ethical_weight", "low")
        if ethical_weight in ["medium", "high"]:
            score -= 1  # higher standard required

        # Research intent — zyada depth chahiye
        if intent_state == "research":
            score -= 1  # stricter standard

        # Execution intent — accuracy critical
        if intent_state == "execution":
            score -= 1

        # ====================================================
        # STEP 4: CONFIDENCE ASSIGN KARO
        # ====================================================

        if score >= 5:    confidence = "high"
        elif score >= 3:  confidence = "medium"
        else:             confidence = "low"

        # ====================================================
        # STEP 5: RETRY DECISION — BILLING CONFIG SE
        # Blueprint: "Public vs Jarvis me behavior alag rakhta hai"
        # Yeh billing mein define hota hai — brain mein tier nahi
        # ====================================================
        
        retry_enabled = config.get("meta_retry_enabled", False)
        threshold     = config.get("meta_confidence_threshold", "low")

        retry = False

        if retry_enabled:
            if threshold == "medium" and confidence != "high":
                # Enterprise + Jarvis — medium ya low pe bhi retry
                retry = True
            elif threshold == "low" and confidence == "low":
                # Free, Paid, Ultra, Business — sirf clearly galat pe retry
                retry = True
        # ====================================================
        # STEP 6: JARVIS EXTRA — SELF-CRITICAL MODE
        # Jarvis ko sabse zyada strict hona chahiye
        # Even "medium" confidence pe extra check
        # ====================================================
        self_critical = config.get("meta_self_critical", False)
        
        if self_critical and confidence == "medium" and retry_enabled:
            # Jarvis medium pe bhi retry karta hai — highest standard
            retry = True 
        # ====================================================
        # STEP 7: INTENT-AWARE OVERRIDE
        # Research + incomplete — retry zaroori agar allowed
        # ====================================================
        if retry_enabled and is_incomplete and intent_state in ["research", "execution"]:
            retry = True

           

        # ====================================================
        # RETURN — downstream format preserve
        # meta["confidence"] → cognitive_profile + world_state
        # meta["retry"]      → llm_generate(retry_prompt)
        # meta["signals"]    → Phase 6A AlignmentFineTuner ke liye
        # ====================================================
        return {
            "confidence": confidence,
            "retry": retry,
            "signals": {
                "length": length,
                "has_structure": has_structure,
                "has_reasoning": has_reasoning,
                "is_incomplete": is_incomplete,
                "score": score,
                "ethical_weight": ethical_weight
            }
        }
        

# =========================
# PHASE 2.4: INTENT STATE ENGINE
# =========================

class IntentStateEngine:

    def detect(self, question: str) -> str:
        """
        Fallback single-intent detection (legacy / safety)
        """
        q = question.lower()

        if any(k in q for k in ["research", "study", "analyze", "compare", "why"]):
            return "research"
        if any(k in q for k in ["do", "build", "create", "execute", "run"]):
            return "execution"
        if any(k in q for k in ["how", "what", "explain", "define"]):
            return "information"

        return "general"

    # ==================================================
    # 🔥 NEW METHOD — LAYER 1 AWARE INTENT DETECTION
    # ==================================================
    def detect_from_layer1(self, layer1_bundle: dict) -> str:
        """
        Uses Layer-1 intent decomposition output to infer
        the dominant intent state.

        Input example:
        layer1_bundle = {
            "intent_type": "mixed",
            "sub_goals": [...],
            "domains": [...],
            "required_depth": "deep"
        }
        """

        if not layer1_bundle:
            return "general"

        intent_type = layer1_bundle.get("intent_type", "general")
        sub_goals = layer1_bundle.get("sub_goals", [])
        required_depth = layer1_bundle.get("required_depth", "normal")

        # 1️⃣ Direct mapping (strongest signal)
        if intent_type in ["research"]:
            return "research"
        if intent_type in ["invention"]:
            return "invention"
        if intent_type in ["planning"]:
            return "planning"
        if intent_type in ["execution", "procedural"]:
            return "execution"
        if intent_type in ["ethical", "philosophical"]:
            return "analysis"
        if intent_type in ["mixed"]:
            if required_depth in ["deep", "very_deep"]:
                return "research"
            return "information"
        if intent_type in ["conceptual", "factual"]:
            return "information"
        #--------------------------
        if intent_type in ["research", "analysis"]:
            return "research"

        if intent_type in ["execution", "action"]:
            return "execution"

        # 2️⃣ Multi-intent → research bias
        if intent_type == "mixed":
            if required_depth in ["deep", "very_deep"]:
                return "research"
            return "information"

        # 3️⃣ Sub-goal heuristic
        for goal in sub_goals:
            g = goal.lower()
            if any(k in g for k in ["compare", "evaluate", "analyze"]):
                return "research"
            if any(k in g for k in ["build", "implement", "execute"]):
                return "execution"

        # 4️⃣ Safe default
        return "information"


# =========================
# PHASE 2.7: SAFETY / CONSTRAINT LAYER
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
        import json

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
    def classify(self, question: str) -> str:
        q = question.lower()

        if any(x in q for x in ["how", "steps", "process", "method"]):
            return "procedural"
        if any(x in q for x in ["who", "when", "where", "date"]):
            return "factual"
        if any(x in q for x in ["why", "explain", "theory", "philosophy"]):
            return "conceptual"

        return "general"


# -------- Phase 3.1 : Source Priority Resolution --------
class SourcePriorityResolver:
    def resolve(self, category: str) -> dict:
        if category == "factual":
            return {"memory": True, "retrieval": True, "reasoning": False}

        if category == "procedural":
            return {"memory": False, "retrieval": True, "reasoning": True}

        if category == "conceptual":
            return {"memory": True, "retrieval": False, "reasoning": True}

        return {"memory": True, "retrieval": True, "reasoning": True}


# -------- Phase 3.2 : Confidence Gating --------
class ConfidenceGate:
    def apply(self, routing: dict, memory_score: float) -> dict:
        if memory_score < 0.4:
            routing["memory"] = False
            routing["reasoning"] = True
        return routing
# -------- Phase 3.4 : Ambiguity Detection --------
class AmbiguityDetector:
    def detect(self, question: str) -> bool:
        vague_terms = ["something", "things", "stuff", "about", "etc"]
        q = question.lower()
        return any(term in q for term in vague_terms)
# -------- Phase 3.5 : Source Conflict Resolution --------
class SourceConflictResolver:
    def resolve(self, routing: dict) -> dict:
        if routing["memory"] and routing["retrieval"]:
            routing["reasoning"] = True
        return routing
# -------- Phase 3.6 : Hallucination Guard --------
class HallucinationGuard:
    def apply(self, routing: dict, confidence: float) -> dict:
        if routing["category"] == "factual" and confidence < 0.3:
            routing["memory"] = False
            routing["reasoning"] = False
            routing["retrieval"] = True
        return routing


# -------- Phase 3.3 : Final Knowledge Router --------
class KnowledgeRouter:
    def route(self, question: str, memory_score: float = 0.5, cognitive_profile: dict = None) -> dict:
        classifier = KnowledgeSourceClassifier()
        resolver = SourcePriorityResolver()
        gate = ConfidenceGate()

        category = classifier.classify(question)
        routing = resolver.resolve(category)
        routing = gate.apply(routing, memory_score)

        return {
            "use_memory": routing["memory"],
            "use_retrieval": routing["retrieval"],
            "use_reasoning": routing["reasoning"],
            "category": category,
            "confidence": memory_score
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
    # INTENT PROTOTYPES — meaning definitions, not keywords
    # Blueprint: "concept-based" — ye centers hain embedding
    # space mein. Jo query in centers ke paas ho = woh intent.
    # ─────────────────────────────────────────────────────
    _INTENT_PROTOTYPES = {
        "factual": (
            "What is the exact fact, date, name, number, or definition? "
            "I need a specific piece of verified information."
        ),
        "conceptual": (
            "Explain this concept deeply. Help me understand "
            "the theory, principle, or underlying idea."
        ),
        "procedural": (
            "What are the steps? How do I do this? "
            "Process, method, instructions to accomplish a task."
        ),
        "research": (
            "Deep investigation needed. Multiple sources, comparison, "
            "multi-layer analysis, synthesis across perspectives."
        ),
        "invention": (
            "Design, create, or invent something new. "
            "Build a new technology, reconstruct ancient design, "
            "innovate using science."
        ),
        "planning": (
            "Create a strategy, roadmap, or multi-step plan. "
            "Goal-oriented approach, Chanakya-style strategy."
        ),
        "ethical": (
            "What is right or wrong? Moral evaluation, dharma, "
            "duty, karma, justice, ethical judgment."
        ),
        "philosophical": (
            "Deeper meaning, truth, ancient wisdom, consciousness, "
            "soul, existence, reality, vedic insight."
        ),
        "conversation": (
            "Casual chat, general talk, no specific task needed, "
            "simple greeting or discussion."
        ),
    }

    # ─────────────────────────────────────────────────────
    # DOMAIN PROTOTYPES — Blueprint Phase 1.6:
    # "Scriptures, Science, Ethics, Philosophy, Technology"
    # Semantic definitions, not keyword lists
    # ─────────────────────────────────────────────────────
    _DOMAIN_PROTOTYPES = {
        "scriptural": (
            "Vedas, Upanishads, Bhagavad Gita, Ramayana, Mahabharata, "
            "Puranas, Hindu scriptures, Sanskrit texts, Vedic knowledge, "
            "Viman Shastra, Arthashastra, Sushruta Samhita, temple wisdom, "
            "mantra, yantra, ancient Bharatiya knowledge."
        ),
        "scientific": (
            "Physics, chemistry, biology, mathematics, quantum mechanics, "
            "energy systems, electromagnetic, acoustic resonance, "
            "scientific laws, experiments, natural phenomena."
        ),
        "technology": (
            "Engineering, machines, devices, propulsion systems, alloys, "
            "materials science, aircraft design, prototype construction, "
            "artificial intelligence, software, hardware, innovation."
        ),
        "ethics": (
            "Moral principles, dharma, right and wrong, duty, karma, "
            "justice, truth, ethical decisions, values, responsibility."
        ),
        "philosophy": (
            "Consciousness, soul, atma, brahman, existence, reality, "
            "Advaita Vedanta, Samkhya, yoga philosophy, enlightenment, "
            "metaphysics, ontology, meaning of life."
        ),
        "history": (
            "Ancient civilizations, historical events, cultural heritage, "
            "Bharatiya tradition, ancient kingdoms, warriors, historical figures."
        ),
        "statecraft": (
            "Governance, politics, Arthashastra, Chanakya, leadership, "
            "policy, administration, power, diplomacy, state management."
        ),
        "ayurveda": (
            "Ayurvedic medicine, herbs, natural healing, body constitution, "
            "Charaka, Sushruta, traditional Indian medicine, wellness."
        ),
    }

    def __init__(self):
        # Pre-compute ALL prototype embeddings once at startup
        # Blueprint: "concept-based" — ye fixed concept centers hain
        self._intent_embs = {
            k: _EMBEDDER.encode(v)
            for k, v in self._INTENT_PROTOTYPES.items()
        }
        self._domain_embs = {
            k: _EMBEDDER.encode(v)
            for k, v in self._DOMAIN_PROTOTYPES.items()
        }

    # ─────────────────────────────────────────────────────
    def process(self, user_query: str, state: dict,
                config: dict = None, memory_graph=None) -> dict:

        config             = config or {}
        tier               = config.get("tier", "free")
        allow_ancient      = config.get("allow_ancient_tech", False)

        # ════════════════════════════════════
        # PHASE 1.0 — Raw Query Capture
        # Blueprint: "User ne jo bola exact, bina interpretation"
        # ════════════════════════════════════
        raw_query = user_query

        # ════════════════════════════════════
        # PHASE 1.1 — Linguistic Normalization
        # Blueprint: "normalize, noise hatao, grammar perfect karo"
        #
        # spaCy: sentence boundaries, entity spans, POS structure
        # Embedder: multilingual — 50+ languages same embedding space
        # Hindi "kya hai ye" aur English "what is this" —
        # same region in embedding space. No translation needed.
        # ════════════════════════════════════
        doc = _NLP(raw_query)   # spaCy linguistic parse

        # Surface clean — only whitespace/punctuation
        # No word substitution, no dictionary, no translation
        normalized_query = re.sub(r'\s+', ' ', raw_query.strip())

        # Semantic embedding — captures meaning across ALL languages
        query_emb = _EMBEDDER.encode(normalized_query)

        # spaCy se linguistic signals extract karo
        # Grok ka achha idea — entities aur POS use karo
        entities   = [(ent.text, ent.label_) for ent in doc.ents]
        # pos_tags   = [(t.text, t.pos_) for t in doc if not t.is_space]
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        has_question_structure = any(
            t.dep_ in ("nsubj", "attr", "dobj") for t in doc
        )

        logging.info(
            f"[Layer1 Ph1.1] entities={entities} | "
            f"nouns={noun_chunks[:3]} | question={has_question_structure}"
        )

        # ════════════════════════════════════
        # PHASE 1.2 — Intent Type Detection
        # Blueprint: "Thinking style — Factual/Conceptual/Mixed etc."
        # Blueprint: "intent concept-based hoga, keyword-based nahi"
        #
        # Embeddings: query ka meaning → intent prototype ke paas
        # spaCy NER: entities se additional signal
        # ════════════════════════════════════
        intent_scores = {
            intent: _cosine(query_emb, proto_emb)
            for intent, proto_emb in self._intent_embs.items()
        }

        sorted_intents = sorted(
            intent_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_intent,  top_score  = sorted_intents[0]
        sec_intent,  sec_score  = sorted_intents[1]

        # spaCy NER signal — Grok ka insight, embedding se combine
        # PERSON entity + low philosophical score → factual boost
        has_person_entity  = any(l in ("PERSON","PER")  for _, l in entities)
        has_org_entity     = any(l in ("ORG","GPE","LOC") for _, l in entities)

        if has_person_entity and top_intent != "factual":
            intent_scores["factual"] = max(
                intent_scores["factual"], top_score * 0.85
            )
            sorted_intents = sorted(
                intent_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_intent, top_score = sorted_intents[0]
            sec_intent, sec_score = sorted_intents[1]

        # ORG/GPE/LOC entity → research ya factual boost
        # Blueprint: "entity-aware thinking style"
        if has_org_entity:
            intent_scores["research"] = max(intent_scores["research"], top_score * 0.75)
            intent_scores["factual"]  = max(intent_scores["factual"],  top_score * 0.70)
            sorted_intents = sorted(
                intent_scores.items(), key=lambda x: x[1], reverse=True
            )
            top_intent, top_score = sorted_intents[0]
            sec_intent, sec_score = sorted_intents[1]    

        # Mixed threshold — dono intents close hain
        MIXED_GAP = 0.07
        if (top_score - sec_score) < MIXED_GAP and sec_score > 0.28:
            intent_type   = "mixed"
            thinking_type = "mixed"
        else:
            intent_type   = top_intent
            thinking_type = top_intent

        logging.info(
            f"[Layer1 Ph1.2] {top_intent}({top_score:.3f}) vs "
            f"{sec_intent}({sec_score:.3f}) → {intent_type}"
        )

        # ════════════════════════════════════
        # PHASE 1.3 — Goal Decomposition (HEART of Layer 1)
        # Blueprint: "ChatGPT ek question ko multiple invisible
        #              goals mein break karta hai"
        # Blueprint: "Bina Memory Graph ke Intent sirf text hoga"
        #
        # Memory Graph: concept activation → hidden goals
        # spaCy: noun chunks → explicit sub-topics
        # ════════════════════════════════════
        sub_goals = []
        # Billing flags — Layer 1 inhe respect karta hai
        # Blueprint: "Billing ka kaam: base power inject karna"
        deep_reasoning   = config.get("deep_reasoning",        False)
        use_emergent     = config.get("use_emergent_concepts",  False)
        query_complexity = config.get("query_complexity",       "low")
        
        # Blueprint: "PUBLIC: Safe, Light, Shallow vs JARVIS: Heavy, deep"
        _complexity_to_max_goals = {
            "low":      2,   # free
            "normal":   3,   # paid
            "high":     5,   # ultra_paid / business / enterprise
            "very_high": 6,  # enterprise
            "maximum":  8,   # jarvis
        }
        max_goals = _complexity_to_max_goals.get(query_complexity, 2)
        # Memory Graph — primary source (Blueprint aligned)
        if memory_graph is not None and use_emergent:
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

        # spaCy noun chunks — Grok ka insight
        # Har noun chunk ek potential sub-goal hai
        if noun_chunks and len(sub_goals) < max_goals:
            for chunk in noun_chunks[:max_goals]:
                goal_candidate = f"investigate_{chunk.replace(' ', '_').lower()}"
                if goal_candidate not in sub_goals:
                    sub_goals.append(goal_candidate)

        # Intent-based semantic fallback (memory graph empty hone par)
        if len(sub_goals) < 2:
            _semantic_goals = {
                "factual":       ["retrieve_exact_information",
                                  "verify_accuracy", "find_source"],
                "conceptual":    ["understand_core_concept",
                                  "find_related_concepts",
                                  "build_mental_model"],
                "procedural":    ["identify_steps", "find_dependencies",
                                  "validate_sequence"],
                "research":      ["deep_investigation",
                                  "multi_source_synthesis",
                                  "compare_perspectives", "find_evidence"],
                "invention":     ["decode_ancient_design",
                                  "map_to_modern_science",
                                  "propose_reconstruction",
                                  "design_experiment"],
                "planning":      ["define_objective", "strategic_analysis",
                                  "multi_step_plan", "risk_assessment"],
                "ethical":       ["ethical_evaluation", "dharma_check",
                                  "identify_consequences"],
                "philosophical": ["find_deeper_meaning",
                                  "vedic_wisdom_connect", "modern_parallel"],
                "conversation":  ["understand_context", "engage_naturally"],
                "mixed":         ["multi_angle_analysis",
                                  "identify_primary_intent",
                                  "synthesize_answer"],
            }
            sub_goals = _semantic_goals.get(
                intent_type, ["understand_user_intent"]
            )
        # Billing cap — free max 2 goals, Jarvis max 8
        sub_goals = sub_goals[:max_goals]
        # ════════════════════════════════════
        # PHASE 1.4 — Query Expansion
        # Blueprint: "ChatGPT internally expanded semantic query banata hai"
        #
        # spaCy entities — Grok ka insight (entities = important concepts)
        # Memory graph neighbors — semantic expansion
        # ════════════════════════════════════
        expanded_queries = [raw_query]

        # spaCy entities se expand — Grok ka valid insight
        if entities:
            entity_texts = [e[0] for e in entities]
            entity_variant = f"{raw_query} {' '.join(entity_texts[:3])}"
            if entity_variant.strip() != raw_query.strip():
                expanded_queries.append(entity_variant)

        # Memory graph semantic neighbors
        if memory_graph is not None and use_emergent:
            try:
                neighbors = memory_graph.get_similar_concepts(
                    query_emb.tolist(), top_k=3
                )
                for n in neighbors:
                    concept = n.get("concept", "")
                    if concept and concept.lower() not in raw_query.lower():
                        expanded_queries.append(f"{raw_query} {concept}")
            except Exception:
                pass

        # Intent-aware semantic variants (meaning-based, not template)
        _semantic_variants = {
            "research":      [f"{raw_query} evidence analysis perspectives"],
            "invention":     [f"{raw_query} ancient scientific principle modern equivalent"],
            "philosophical": [f"{raw_query} vedic wisdom deeper meaning"],
            "planning":      [f"{raw_query} strategic approach step by step"],
        }
        for v in _semantic_variants.get(intent_type, []):
            if v not in expanded_queries:
                expanded_queries.append(v)

        # ════════════════════════════════════
        # PHASE 1.5 — Reasoning Depth Estimation
        # Blueprint: "Short answer chalega ya heavy multi-layer reasoning?"
        #
        # Embedding confidence + spaCy complexity + goal count
        # Grok ka idea (sentence length) + embedding confidence combine
        # ════════════════════════════════════
        n_sentences     = len(list(doc.sents))
        n_entities      = len(entities)
        goal_count      = len(sub_goals)
        top_confidence  = top_score

        graph_relevance = 0.0
        if memory_graph is not None:
            try:
                graph_relevance = memory_graph.estimate_relevance(raw_query)
            except Exception:
                pass

        # Multi-signal depth decision
        if intent_type in ["invention", "philosophical"] and goal_count >= 3:
            required_depth = "very_deep"
        elif intent_type == "research" or goal_count >= 4:
            required_depth = "deep"
        elif n_sentences >= 3 or n_entities >= 3:
            # Grok ka sentence/entity insight — complex linguistic structure
            required_depth = "deep"
        elif intent_type in ["conceptual", "ethical"] or goal_count >= 2:
            required_depth = "normal"
        elif intent_type == "factual" and top_confidence > 0.65:
            required_depth = "shallow"
        elif graph_relevance > 0.7:
            required_depth = "deep"
        else:
            required_depth = "normal"
        if not deep_reasoning:
            # PUBLIC tiers — max "normal", kabhi deep/very_deep nahi
            if required_depth in ["deep", "very_deep"]:
                required_depth = "normal"
        elif deep_reasoning and required_depth in ["normal", "shallow"]:
            # deep_reasoning True = Jarvis/Enterprise — minimum deep
            required_depth = "deep"
        logging.info(
            f"[Layer1 Ph1.5] depth={required_depth} | "
            f"sents={n_sentences} | ents={n_entities} | "
            f"goals={goal_count} | graph={graph_relevance:.2f}"
        )

        # ════════════════════════════════════
        # PHASE 1.6 — Knowledge Domain Mapping
        # Blueprint: "Scriptures, Science, Ethics, Philosophy, Technology"
        #
        # Embeddings: pure semantic — no keyword lists
        # spaCy entities: domain-relevant named entities detect karo
        # ════════════════════════════════════
        DOMAIN_THRESH = 0.35

        domain_scores = {
            d: _cosine(query_emb, proto_emb)
            for d, proto_emb in self._domain_embs.items()
        }
        domains = [
            d for d, score in domain_scores.items()
            if score >= DOMAIN_THRESH
        ]
        domains = sorted(domains, key=lambda d: domain_scores[d], reverse=True)

        # Billing gate — config se (blueprint: "allow_ancient_tech Jarvis only")
        if not allow_ancient:
            domains = [
                d for d in domains
                if d not in ["scriptural", "ayurveda"]
            ]

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
            "intent_type":       intent_type,
            "thinking_type":     thinking_type,
            "intent_scores":     intent_scores,
            "sub_goals":         sub_goals,
            "expanded_queries":  expanded_queries,
            "required_depth":    required_depth,
            "domains":           domains,
            "domain_scores":     domain_scores,
            "reasoning_plan":    sub_goals[:3],
            # spaCy signals — Layer 2, 3 ke liye
            "entities":          entities,
            "noun_chunks":       noun_chunks,
            "n_sentences":       n_sentences,
        }

        state["layer1_intent_bundle"] = intent_bundle
        return state
    

class MemoryAwarePruner:
    """
    Prunes expanded queries using memory relevance.
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def prune(self, queries: list[str], memory_graph) -> list[str]:
        pruned = []
        for q in queries:
            score = memory_graph.estimate_relevance(q)
            if score >= self.threshold:
                pruned.append(q)
        return pruned

#--------LAYER 2 KA PHASE 2.5 MEMORY AWARE QUERY PRUNING (LAYER 4 HOOK)
class MemoryAwareQueryPruner:
    def prune(self, queries, memory_graph, intent_state):
        """
        Remove redundant or low-value queries
        based on:
        - memory similarity
        - intent criticality
        """
        pruned = []
        seen = set()

        for q in queries:
            key = q.lower().strip()
            if key in seen:
                continue
            seen.add(key)

            score = memory_graph.estimate_relevance(q)

            # Blueprint: "kya ye knowledge already store hai? drop karo"
            if score > 0.85:
                # Already strongly in memory — skip, waste hogi
                continue

            # Blueprint: "kuch deeper" — low memory = naya topic = deeper explore karo
            if score < 0.2 and intent_state in ["research", "invention"]:
                pruned.append(q + " — deep investigation required")
            else:
                pruned.append(q)

        return pruned if pruned else queries  # fallback: kuch to dena hai

#================================================
#==========LAYER 2 : ADAPTIVE QUERY EXPANSION(DYNAMICS)==========
#================================================
# .......Phase 2.1 : INTENT-wise QUERY BRANCHING.........
class IntentQueryBrancher:
    def branch(self, intent_state, sub_goals):
        branches = []

        if intent_state == "research":
            for goal in sub_goals:
                branches.append({"type": "exploratory", "goal": goal})

        elif intent_state == "execution":
            branches.append({"type": "procedural", "goal": "step-by-step implementation"})
            branches.append({"type": "causal", "goal": "why each step matters"})

        elif intent_state == "invention":
            branches.append({"type": "hypothetical", "goal": "design possibilities"})
            branches.append({"type": "comparative", "goal": "ancient vs modern equivalent"})
            branches.append({"type": "causal", "goal": "underlying scientific principle"})

        elif intent_state == "planning":
            branches.append({"type": "declarative", "goal": "objective definition"})
            branches.append({"type": "causal", "goal": "strategic dependencies"})

        elif intent_state == "analysis":
            branches.append({"type": "comparative", "goal": "multi-angle evaluation"})
            branches.append({"type": "causal", "goal": "root cause identification"})

        elif intent_state == "information":
            branches.append({"type": "declarative", "goal": "exact answer"})
            branches.append({"type": "exploratory", "goal": "related context"})

        else:
            branches.append({"type": "exploratory", "goal": "overview"})

        return branches
#.............. Phase 2.2 : QUERY GRANULARITY DECISION ..........
class QueryGranularityDecider:
    def decide(self, intent_state, required_depth):
        if required_depth in ["deep", "very_deep"] or intent_state in ["research", "invention"]:
            return "fine"
        elif intent_state in ["philosophical", "analysis"]:
            return "abstract"
        elif required_depth == "shallow" or intent_state == "information":
            return "narrow"
        return "normal"

#.................. Phase 2.3 : DYNAMIC QUERY SHAPE GENERATOR ..........
class QueryShapeGenerator:
    def generate(self, base_question, branch, granularity):
        t = branch["type"]
        goal = branch.get("goal", "")

        if t == "declarative":
            return f"What exactly is: {base_question}"
        if t == "exploratory":
            return f"{base_question} — explore: {goal}"
        if t == "hypothetical":
            return f"If we were to {goal}, how would {base_question} work?"
        if t == "comparative":
            return f"Compare {goal} in context of: {base_question}"
        if t == "causal":
            return f"Why and how does {goal} relate to: {base_question}"
        if t == "procedural":
            return f"Step-by-step: {goal} for {base_question}"

        return base_question
#...............Phase 2.4 : ABSTRACTION LEVEL MODULATOR ..........
class AbstractionModulator:
    def adjust(self, query, granularity):
        if granularity == "fine":
            # Concrete level — facts aur data
            return query + " with specific data, measurements and verified facts"
        if granularity == "abstract":
            # Meta level — ethics, philosophy
            return f"From a philosophical and ethical perspective: {query}"
        if granularity == "narrow":
            # Conceptual level — theory
            return f"Core concept and theory behind: {query}"
        return query


#..........Phase 2.6 : QUERY PRIORITY & BUDGET ALLOCATION ..........
class QueryBudgetAllocator:
    def allocate(self, queries, cognitive_profile):
        budget = cognitive_profile.get("max_docs", 6)
        # Blueprint: "intent importance" — causal/hypothetical pehle, general baad mein
        priority_keywords = ["why", "how", "deep", "investigate", "principle", "vedic", "ancient"]

        def priority_score(q):
            q_lower = q.lower()
            return sum(1 for kw in priority_keywords if kw in q_lower)

        prioritized = sorted(queries, key=priority_score, reverse=True)
        return prioritized[:budget]
#............Phase 2.7 : FINAL QUERY BUNDLE OUTPUT..........
class AdaptiveQueryExpansionEngine:
    def run(
        self,
        question,
        layer1_bundle,
        intent_state,
        cognitive_profile,
        memory_graph
    ):
        brancher = IntentQueryBrancher()
        granularity_decider = QueryGranularityDecider()
        shape_gen = QueryShapeGenerator()
        abstraction = AbstractionModulator()
        pruner = MemoryAwarePruner()
        allocator = QueryBudgetAllocator()

        branches = brancher.branch(
            intent_state,
            layer1_bundle.get("sub_goals", [])
        )

        granularity = granularity_decider.decide(
            intent_state,
            layer1_bundle.get("required_depth", "normal")
        )

        queries = []
        for b in branches:
            q = shape_gen.generate(question, b, granularity)
            q = abstraction.adjust(q, granularity)
            queries.append(q)

        queries = pruner.prune(queries, memory_graph)
        queries = allocator.allocate(queries, cognitive_profile)
        
        queries = MemoryAwareQueryPruner().prune(
                    queries,
                    memory_graph,
                    intent_state
                )

        # query_complexity — billing se aata hai, retrieval depth decide karta hai
        complexity = cognitive_profile.get("query_complexity", "normal")

        if complexity == "low":
            # Free tier — sirf primary query, koi expansion nahi
            queries = queries[:1] if isinstance(queries, list) else queries

        elif complexity == "normal":
            # Paid — basic expansion, max 3 queries
            if isinstance(queries, list):
                queries = queries[:3]

        elif complexity == "high":
            # Ultra/Business — full expansion, saare branches
            pass  # sab queries use karo
        
        elif complexity == "very_high":
            # Enterprise — saari queries + har query ka ek semantic variant add karo
            if isinstance(queries, list):
                expanded = []
                for q in queries:
                    expanded.append(q)
                    expanded.append(f"Elaborate on: {q}")
                queries = expanded

        elif complexity == "maximum":
            # Jarvis — saari queries + semantic + abstract + vedic variant
            if isinstance(queries, list):
                expanded = []
                for q in queries:
                    expanded.append(q)
                    expanded.append(f"Elaborate on: {q}")
                    expanded.append(f"From Vedic-scientific perspective: {q}")
                    expanded.append(f"Hidden assumptions in: {q}")
                queries = expanded

        logging.info(f"[Query Complexity: {complexity}] Final query count: {len(queries) if isinstance(queries, list) else 'dict'}")
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
    tier = config["tier"]  
    # mode = "jarvis" if tier == "jarvis" else "public"


    training_result = training_controller.maybe_train(
        dataset_path="./training/datasets/sample.jsonl"
    )

    logging.info(f"[PHASE 6 TRAINING] → {training_result}")

    # Initialize the language model
          

    chain = create_chain(llm)

    question = "How to report The Prince ?"

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
    # # Layer-1 Hard Guarantees (Production Safety)
    # layer1_bundle.setdefault("normalized_query", question)           #parmanent
    # layer1_bundle.setdefault("thinking_type", "mixed")                                  delete krna hai
    # layer1_bundle.setdefault("reasoning_plan", [])

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
        intent=intent_state,
        domains=layer1_bundle.get("domains", []),
        required_depth=layer1_bundle.get("required_depth", "normal")
    )
    
    logging.info(f"[Cognitive Route] → {cognitive_route}")
    #-------------------------------------------------------------
    world_state = {
    "domain": None,
    "human_factor": False,
    "ethical_weight": "low"
    }

    #--------------------------------------------------------------
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
    cognitive_profile.setdefault("max_docs", 6)
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
        intent_state=intent_state,
        cognitive_profile=cognitive_profile,
        memory_graph=memory_graph   # Layer 4 hook
    )
    #---------------------------------------------------
    # 🔒 Layer-2 Production Lock
    if not isinstance(adaptive_queries, (dict, list)):                     #parmanent
        raise RuntimeError("Layer-2 output corrupted. Blueprint violation.")

    #-----------------------------------------------
    
    logging.info(f"[Adaptive Queries] → {adaptive_queries}")
    
    #--------------------------------------------------------------------------
       

    # ===== Layer-2 → Router Bridge (Blueprint Compliant Fix) =====
    try:
        # Agar adaptive_queries Dictionary hai (Best for AGI Blueprint)
        adaptive_query_text = (
            " ".join(
                q for group in adaptive_queries.values()
                for q in group
            )
            if adaptive_queries else ""
        )

    except AttributeError:
        # Fallback: Agar engine ne sirf ek simple List bheji hai
        adaptive_query_text = " ".join(adaptive_queries)
    
    logging.info(f"Bridge Active: Combined Query for Layer-3 -> {adaptive_query_text}")

    # # ===== Layer-2 → Router Bridge =====
    # adaptive_query_text = " ".join(
    #     q for group in adaptive_queries.values()
    #     for q in group
    # )
    # # ===== Layer-2 Adaptive Feedback =====
    # if sum(len(v) for v in adaptive_queries.values()) > 6:
    #     cognitive_profile["deep_reasoning"] = True
    #     cognitive_profile["query_complexity"] = "high"
    
    query_count = sum(len(v) for v in adaptive_queries.values()) if isinstance(adaptive_queries, dict) else len(adaptive_queries)
    
    if query_count > 6 and not cognitive_profile.get("deep_reasoning"):
        cognitive_profile["deep_reasoning"] = True
        cognitive_profile["query_complexity"] = "high"


    # if query_count > 6:
    #     cognitive_profile["deep_reasoning"] = True
    #     cognitive_profile["query_complexity"] = "high"

    # 🔁 Layer-2 → Cognitive Sync
    if cognitive_profile.get("deep_reasoning"):
        cognitive_profile["max_docs"] = max(
            cognitive_profile.get("max_docs", 6), 8
        )

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

 
    # ===== Phase 3.6 : FINAL KNOWLEDGE ROUTING (ONLY CALL) =====
    router3 = KnowledgeRouter()
    knowledge_route = router3.route(
        question=(
            mutated_question + " " +
            " ".join(layer1_bundle.get("expanded_queries", [])) + " " + adaptive_query_text  # 👈 VERY IMPORTANT
        ),
        memory_score=cognitive_profile.get("confidence", 0.6),
        cognitive_profile=cognitive_profile
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
        memory = memory_lookup(vector_db, question)
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
            k=10
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
