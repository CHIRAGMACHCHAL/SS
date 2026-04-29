# usage/usage_manager.py

import os
from datetime import datetime
import asyncpg
import redis.asyncio as redis
import pytz

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

pg_pool = None
redis_client = None

async def init_usage():
    global pg_pool, redis_client
    pg_pool = await asyncpg.create_pool(DATABASE_URL)
    redis_client = await redis.from_url(REDIS_URL)


# =========================
# Helper
# =========================
async def get_user_timezone(email: str) -> str:
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT timezone FROM users WHERE email=$1",
            email
        )
        return row["timezone"] if row else "UTC"


def get_today_in_timezone(tz_str):
    tz = pytz.timezone(tz_str)
    return datetime.now(tz).strftime("%Y-%m-%d")


# =========================
# GET USAGE
# =========================
async def get_usage(email: str) -> int:
    tz = await get_user_timezone(email)
    today = get_today_in_timezone(tz)

    redis_key = f"usage:{email}:{today}"

    # 🔥 Redis first (fast)
    usage = await redis_client.get(redis_key)
    if usage:
        return int(usage)

    # fallback to DB
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT count FROM usage WHERE email=$1 AND date=$2",
            email, today
        )
        return row["count"] if row else 0


# =========================
# INCREMENT
# =========================
async def increment_usage(email: str):
    tz = await get_user_timezone(email)
    today = get_today_in_timezone(tz)

    redis_key = f"usage:{email}:{today}"

    # Redis increment
    new_val = await redis_client.incr(redis_key)

    # expire at midnight (auto reset)
    await redis_client.expire(redis_key, 86400)

    # Async DB sync (non-blocking)
    async with pg_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO usage (email, date, count)
            VALUES ($1, $2, $3)
            ON CONFLICT (email, date)
            DO UPDATE SET count = usage.count + 1
            """,
            email, today, 1
        )

# iss code ka yhi table hai 
# CREATE TABLE usage (
#     email TEXT,
#     date DATE,
#     count INT DEFAULT 0,
#     PRIMARY KEY (email, date)
# );        



#  ye pichhle ka code hai jo llm_engine mein tha, usme usage check aur increment karna tha, ab usko yaha se import karke use karenge
# # ✅ FINAL FIX (Timezone-aware usage)
# # Step 1: user timezone store करो (DB में)

# # Table:

# ALTER TABLE users ADD COLUMN timezone TEXT DEFAULT 'UTC';

# # Example:

# # India user → Asia/Kolkata
# # US user → America/New_York

# 1. conversation_memory.py
# 2. graph_sync.py
# 3. ingestion.py
# 4. storage.py
# 5. vector_store.py