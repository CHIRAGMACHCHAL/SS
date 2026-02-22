# memory/graph_sync.py
# Layer 4: Memory Graph (Understanding Layer/Engine)
# Blueprint ke hisaab se: "Memory Graph = samjhna"

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set
import asyncpg
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from .storage import MemoryStorageLayer  # Storage layer se config aur connections ke liye hook

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

class MemoryGraphNode:
    """Ek concept node - text, embedding, metadata"""
    def __init__(self, concept: str, embedding: List[float], metadata: Dict = None):
        self.id = str(uuid.uuid4())
        self.concept = concept.lower().strip()
        self.embedding = np.array(embedding)
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow().isoformat()
        self.last_updated = self.created_at
        self.strength = 1.0  # activation strength (training se badhega)

    def to_dict(self):
        return {
            "id": self.id,
            "concept": self.concept,
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "strength": self.strength
        }


class MemoryGraphEdge:
    """Concept ke beech relation"""
    def __init__(self, source_id: str, target_id: str, weight: float = 1.0, relation_type: str = "co-occurrence"):
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.relation_type = relation_type
        self.last_seen = datetime.utcnow().isoformat()


class MemoryGraph:
    """
    Layer 4 Core: Persistent, async, tier-aware Memory Graph
    - Concepts + Relations store karta hai
    - Emergent se shuru hota hai → training se evolve hota hai
    - Jarvis mein deep + identity reinforcement
    """
    def __init__(self):
        self.pg_pool = None
        self.redis = None
        self.nodes: Dict[str, MemoryGraphNode] = {}      # in-memory cache
        self.edges: List[MemoryGraphEdge] = []           # in-memory cache
        self.storage = MemoryStorageLayer()              # Storage layer se config aur connections ke liye hook

    async def init_connections(self):
        """PostgreSQL + Redis connect (storage layer se aa sakta hai)"""
        self.pg_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
        self.redis = await redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    async def _get_or_create_node(self, concept: str, email: str = None) -> MemoryGraphNode:
        """Concept node banao ya fetch karo"""
        concept = concept.lower().strip()
        
        if email is None:
            email = "chirag@example.com"  # default fallback
            config = await self.storage.get_storage_config(email)  # storage se config le lo
            tier = config["tier"]

        else:
            config = await self.storage.get_storage_config(email)
            tier = config["tier"] 
        cache_key = f"node:{tier}:{concept}"    

        # Redis se fast check
        cached = await self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            node = MemoryGraphNode(concept, data["embedding"], data["metadata"])
            node.id = data["id"]
            node.strength = data["strength"]
            return node

        # PostgreSQL se check
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memory_graph_nodes WHERE concept = $1 AND tier = $2",
                concept, tier
            )
            if row:
                node = MemoryGraphNode(concept, row["embedding"], row["metadata"])
                node.id = row["id"]
                node.strength = row["strength"]
            else:
                # Naya node banao
                embedding = embedder.encode(concept).tolist()
                node = MemoryGraphNode(concept, embedding, {"tier": tier})
                await conn.execute(
                    """
                    INSERT INTO memory_graph_nodes (id, concept, embedding, metadata, strength, tier)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    node.id, node.concept, node.embedding.tolist(), node.metadata, node.strength, tier
                )

        # Cache kar do
        await self.redis.set(cache_key, json.dumps(node.to_dict()), ex=3600)  # 1 hour
        self.nodes[node.id] = node
        return node

    async def _add_or_update_edge(self, source_id: str, target_id: str, weight: float = 1.0, relation_type: str = "co-occurrence"):
        """Relation add ya update"""
        async with self.pg_pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT weight FROM memory_graph_edges WHERE source_id = $1 AND target_id = $2",
                source_id, target_id
            )
            if existing:
                new_weight = min(5.0, existing["weight"] + weight)  # cap at 5
                await conn.execute(
                    "UPDATE memory_graph_edges SET weight = $1, last_seen = NOW() WHERE source_id = $2 AND target_id = $3",
                    new_weight, source_id, target_id
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO memory_graph_edges (source_id, target_id, weight, relation_type, last_seen)
                    VALUES ($1, $2, $3, $4, NOW())
                    """,
                    source_id, target_id, weight, relation_type
                )

    async def sync_to_memory_graph(self, question: str, answer: str, email: str = None):
        """
        Phase 6 + Conversation hook: Question + Answer se concepts + relations banao
        """
        if email is None:
            email = "chirag@example.com"
        config = await self.storage.get_storage_config(email)
        tier = config["tier"]

        # Text combine karo
        text = f"{question} {answer}"
        words = [w.lower().strip() for w in text.split() if len(w) > 4]

        # Top concepts nikaalo (emergent)
        concept_count = defaultdict(int)
        for w in words:
            concept_count[w] += 1

        top_concepts = [c for c, cnt in sorted(concept_count.items(), key=lambda x: x[1], reverse=True)][:8]

        if not top_concepts:
            return

        # Nodes banao ya fetch karo
        nodes = {}
        for concept in top_concepts:
            node = await self._get_or_create_node(concept, tier)
            nodes[concept] = node

        # Edges banao (co-occurrence)
        for i, c1 in enumerate(top_concepts):
            for c2 in top_concepts[i+1:]:
                if c1 != c2:
                    await self._add_or_update_edge(nodes[c1].id, nodes[c2].id, weight=0.8)

        # Jarvis special: Identity reinforcement node
        if tier == "jarvis":
            identity_node = await self._get_or_create_node("jarvis_identity", tier)
            for concept in top_concepts:
                await self._add_or_update_edge(identity_node.id, nodes[concept].id, weight=1.2, relation_type="identity_core")

        print(f"[Layer 4 Sync] → {len(top_concepts)} concepts, {len(top_concepts)*(len(top_concepts)-1)//2} edges added/updated")

    async def expand_query_with_graph(self, query: str, email: str = None, max_related: int = 5) -> str:
        """
        Layer 2 / Layer 3 hook: Query ko graph se expand karo
        """
        if email is None:
            email = "chirag@example.com"
        config = await self.storage.get_storage_config(email)
        tier = config["tier"]

        words = [w.lower().strip() for w in query.split() if len(w) > 4]
        if not words:
            return query

        expanded = [query]

        for word in words[:3]:  # top 3 se shuru
            node = await self._get_or_create_node(word, tier)
            if not node:
                continue

            # Related concepts pull karo
            async with self.pg_pool.acquire() as conn:
                related = await conn.fetch(
                    """
                    SELECT target_id, weight FROM memory_graph_edges 
                    WHERE source_id = $1 ORDER BY weight DESC LIMIT $2
                    """,
                    node.id, max_related
                )

                for row in related:
                    target_node = await conn.fetchrow(
                        "SELECT concept FROM memory_graph_nodes WHERE id = $1",
                        row["target_id"]
                    )
                    if target_node:
                        expanded.append(target_node["concept"])

        return " ".join(set(expanded))  # unique rakho

    async def get_related_concepts(self, concept: str, email: str = None, limit: int = 8) -> List[str]:
        """Debug / Jarvis ke liye"""
        if email is None:
            email = "chirag@example.com"
        config = await self.storage.get_storage_config(email)
        tier = config["tier"]

        node = await self._get_or_create_node(concept, tier)
        if not node:
            return []

        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT n.concept, e.weight 
                FROM memory_graph_edges e
                JOIN memory_graph_nodes n ON e.target_id = n.id
                WHERE e.source_id = $1
                ORDER BY e.weight DESC LIMIT $2
                """,
                node.id, limit
            )
            return [r["concept"] for r in rows]

    async def close(self):
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis:
            await self.redis.close()


# #  TABLE JO POSTGRESQL MEIN BANANI HAI (migration ke liye):
# #  TABLE JO DATABASE ME CHAHIYE (POSTGRESQL):

# CREATE TABLE memory_graph_nodes (
#     id UUID PRIMARY KEY,
#     concept TEXT NOT NULL,
#     embedding VECTOR(768),          -- mpnet-base-v2 = 768 dim
#     metadata JSONB,
#     strength FLOAT DEFAULT 1.0,
#     tier TEXT DEFAULT 'public',
#     created_at TIMESTAMP,
#     last_updated TIMESTAMP
# );

# CREATE TABLE memory_graph_edges (
#     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
#     source_id UUID REFERENCES memory_graph_nodes(id),
#     target_id UUID REFERENCES memory_graph_nodes(id),
#     weight FLOAT DEFAULT 1.0,
#     relation_type TEXT DEFAULT 'co-occurrence',
#     last_seen TIMESTAMP DEFAULT NOW()
# );

# CREATE INDEX idx_concept_tier ON memory_graph_nodes(concept, tier);
# CREATE INDEX idx_edge_source ON memory_graph_edges(source_id);