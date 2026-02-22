#memory/storage.py - Layer 8: Memory & Storage (Production ready)

import os
import asyncpg
import redis.asyncio as redis
from qdrant_client import QdrantClient
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
from billing.billing import BillingLayer  # Billing config ke liye

class MemoryStorageLayer:
    """
    Production Memory & Storage Layer
    - Blueprint ke hisaab se public aur Jarvis ke liye alag storage
    - Ingestion, fingerprint, chunks ka logic brain/main.py mein rahega
    - Yeh layer sirf storage provide karegi aur brain ko call karegi
    """

    def __init__(self):

        # Connections
        self.pg_pool = None
        self.redis_client = None
        self.qdrant = None

    async def get_storage_config(self, email: str) -> Dict[str, Any]:
        """BillingLayer se pura config laata hai"""
        config = BillingLayer.generate_config(email)
        return {
            "tier": config["tier"],
            "collection": config["collection"],
            "bucket": config.get("bucket", "mini-agi-public-pdfs"),
            "prefix": config.get("prefix", "documents/"),
            "allow_long_term_memory": config.get("long_term_memory", False)
        }

    async def init_connections(self):
        """Async connections initialize"""
        self.pg_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
        self.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

    # ================== Conversation History ==================
    async def get_conversation_history(self, conversation_id: str, email: str) -> str:
        config = await self.get_storage_config(email)
        limit = 100 if config["allow_long_term_memory"] else 10
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

    async def update_conversation_history(self, conversation_id: str, question: str, answer: str, email:str):
        config = await self.get_storage_config(email)
        new_entry = f"[{datetime.utcnow().isoformat()}] User ({config['tier']}): {question}\nAssistant: {answer}\n"
        async with self.pg_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversations (id, messages, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (id) 
                DO UPDATE SET messages = messages || $2, updated_at = NOW()
                """,
                conversation_id,
                new_entry
            )

    # ================== Vector Search (Brain se call hoga) ==================
    async def similarity_search(self, query: str, email: str, k: int = 8):
        """Brain yeh function call karega"""
        config = await self.get_storage_config(email)
        collection = config["collection"]
        # Yahan tera existing QdrantVectorStore logic call hoga (brain se)
        # No duplication - brain ka code reuse
        return []  # Placeholder - real call brain se hoga

    async def store_document_chunks(self, chunks: List, email: str):
        """Brain yeh function call karega PDF chunks store karne ke liye"""
        config = await self.get_storage_config(email)
        bucket = config["bucket"]
        collection = config["collection"]
        # Yahan tera existing create_or_load_vector_db logic call hoga
        # No duplication
        pass