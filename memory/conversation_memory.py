# memory/conversation_memory.py - Layer 8: Persistent Identity & Understanding Memory

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import asyncpg
import redis.asyncio as redis
from .graph_sync import MemoryGraph  # Layer 4 hook
from llm.llm_engine import generate as llm_generate   # ← yeh add karo
from .storage import MemoryStorageLayer   

class ConversationMemory:
    """
    Layer 8 - Persistent Memory Engine
    - Short-term: current session
    - Long-term: previous conversations + projects (Jarvis only)
    - Tier-aware: public shallow, jarvis deep + identity
    - Phase 6 patches ingest support
    """

    def __init__(self):
        self.storage = MemoryStorageLayer()
        self.graph = MemoryGraph()
        self.pg_pool = None
        self.redis = None
        
    async def init_connections(self):
        """Storage layer se saare connections initialize"""
        await self.storage.init_connections()          # ← yahan se aa raha hai
        await self.graph.init_connections()
        self.pg_pool = self.storage.pg_pool
        self.redis = self.storage.redis_client

    async def get_conversation_history(self, conversation_id: str, email: str = None, limit: int = 10) -> str:
        """Get recent messages (Jarvis mein zyada deep history)"""
        if email is None:
            email = "chirag@example.com"
        config = await self.storage.get_storage_config(email)
        tier = config["tier"]
      
        if tier == "jarvis":
            limit = 100  # Jarvis ko zyada yaad rahega

        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT messages FROM conversations
                WHERE id = $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                conversation_id,
                limit
            )
            if rows:
                return "\n".join(row["messages"] for row in rows)
            return ""

    async def update_conversation(self,
                                 conversation_id: str,
                                 question: str,
                                 answer: str,
                                 email: str = None,
                                 project_context: Optional[str] = None):
        """Update history + sync to Memory Graph + Jarvis identity reinforcement"""
        timestamp = datetime.utcnow().isoformat()
        if email is None:
            email = "chirag@example.com"
        config = await self.storage.get_storage_config(email)
        tier = config["tier"]

        entry = f"[{timestamp}] User ({tier}): {question}\nAssistant: {answer}\n"

        if project_context:
            entry += f"[Project Context]: {project_context}\n"

        # Redis mein short-term cache (fast access)
        await self.redis.append(f"conv:{conversation_id}", entry)

        # PostgreSQL mein long-term persistent storage
        async with self.pg_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (id, messages, tier, updated_at, project_context)
                VALUES ($1, $2, $3, NOW(), $4)
                ON CONFLICT (id) DO UPDATE
                SET messages = messages || $2,
                    updated_at = NOW(),
                    project_context = COALESCE($4, project_context)
                """,
                conversation_id,
                entry,
                tier,
                project_context
            )

        # Layer 4 Memory Graph sync (concept + relation update)
        await self.graph.sync_to_memory_graph(
            question=question,
            answer=answer,
            email=email
        )
        #-----------------------------

        # Jarvis special: identity reinforcement
        if tier == "jarvis":
            await self._reinforce_jarvis_identity(conversation_id, question, answer)

    async def _reinforce_jarvis_identity(self, conv_id: str, question: str, answer: str):
        """Jarvis ko apni identity yaad dilata hai (Tony Stark style)"""
        identity_prompt = f"""
        You are Jarvis - Chirag's long-term Vedic-Scientific partner.
        Previous context: {question}
        Your response: {answer}
        Reinforce your identity in one line only.
        """
        identity_line = llm_generate(identity_prompt)  # short call
        await self.redis.append(f"identity:{conv_id}", f"[Identity] {identity_line}\n")

    async def search_memory(self, query: str, email: str = None, k: int = 5) -> str:
        """Semantic search across history (Jarvis mein zyada deep)"""
        if email is None:
            email = "chirag@example.com"
        config = await self.storage.get_storage_config(email)
        if config["tier"] == "jarvis":
            k = 15  # Jarvis ko zyada context milega

        # Redis se fast short-term
        recent = await self.redis.getrange(f"conv:recent", 0, -1)
        if recent:
            return recent.decode()

        # PostgreSQL se long-term
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT messages FROM conversations
                WHERE messages ILIKE $1
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                f"%{query}%",
                k
            )
            return "\n\n".join(row["messages"] for row in rows) if rows else ""

    async def get_project_context(self, conversation_id: str) -> Optional[str]:
        """Jarvis ke liye project continuity"""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT project_context FROM conversations WHERE id = $1",
                conversation_id
            )
            return row["project_context"] if row else None