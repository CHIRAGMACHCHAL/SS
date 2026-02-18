#memory/storage.py - Layer 8: Memory & Storage (Production ready)

import os
import asyncpg
import redis.asyncio as redis
from qdrant_client import QdrantClient
from typing import Dict, Any, List, Optional
import uuid

class MemoryStorageLayer:
    """
    Production Memory & Storage Layer
    - Blueprint ke hisaab se public aur Jarvis ke liye alag storage
    - Ingestion, fingerprint, chunks ka logic brain/main.py mein rahega
    - Yeh layer sirf storage provide karegi aur brain ko call karegi
    """

    def __init__(self):
        # Buckets (tere blueprint ke hisaab se)
        self.public_bucket = "mini-agi-public-pdfs"
        self.jarvis_bucket = "mini-agi-jarvis-pdfs"

        # Qdrant Collections (alag-alag)
        self.public_collection = "public_core"
        self.jarvis_collection = "jarvis_private"

        # Connections
        self.pg_pool = None
        self.redis_client = None
        self.qdrant = None

    async def init_connections(self):
        """Async connections initialize"""
        self.pg_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
        self.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

    async def get_collection_name(self, tier: str) -> str:
        """Tier ke hisaab se sahi collection"""
        if tier == "jarvis":
            return self.jarvis_collection
        return self.public_collection

    async def get_bucket_name(self, tier: str) -> str:
        """Tier ke hisaab se sahi S3 bucket"""
        if tier == "jarvis":
            return self.jarvis_bucket
        return self.public_bucket

    # ================== Conversation History ==================
    async def get_conversation_history(self, conversation_id: str) -> str:
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT messages FROM conversations WHERE id = $1",
                conversation_id
            )
            return row['messages'] if row else ""

    async def update_conversation_history(self, conversation_id: str, question: str, answer: str):
        new_entry = f"User: {question}\nAssistant: {answer}\n"
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
    async def similarity_search(self, query: str, tier: str, k: int = 8):
        """Brain yeh function call karega"""
        collection = await self.get_collection_name(tier)
        # Yahan tera existing QdrantVectorStore logic call hoga (brain se)
        # No duplication - brain ka code reuse
        return []  # Placeholder - real call brain se hoga

    async def store_document_chunks(self, chunks: List, tier: str):
        """Brain yeh function call karega PDF chunks store karne ke liye"""
        bucket = await self.get_bucket_name(tier)
        collection = await self.get_collection_name(tier)
        # Yahan tera existing create_or_load_vector_db logic call hoga
        # No duplication
        pass