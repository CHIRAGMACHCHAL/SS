# Quick Start Guide - SS Mini AGI

Get up and running with the Vedic Intelligence System in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- Git

## Installation

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/CHIRAGMACHCHAL/SS.git
cd SS

# Run automated setup
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create virtual environment
- Install all dependencies
- Download required models
- Create configuration files
- Set up data directories

### 2. Configure Environment

Edit the `.env` file with your settings:

```bash
# Minimum required settings
DATABASE_URL=postgresql://user:pass@localhost:5432/vedic_agi
QDRANT_URL=http://localhost:6333
```

### 3. Start Required Services

**Option A: Using Docker (Recommended)**

```bash
# Start Qdrant vector database
docker run -d -p 6333:6333 \
  -v ./data/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Start PostgreSQL
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=vedic_agi \
  postgres:14

# Start Redis
docker run -d -p 6379:6379 redis:7
```

**Option B: Local Installation**

Install PostgreSQL, Redis, and run Qdrant via Docker as shown above.

### 4. Run Your First Query

```bash
# Activate virtual environment
source venv/bin/activate

# Run interactive demo
python demo.py interactive
```

## Quick Examples

### Example 1: Basic Query

```python
import asyncio
from brain.main import main as agi_brain
from billing.billing import BillingLayer
from memory.vector_store import VectorStoreManager
from memory.graph_sync import MemoryGraph
from memory.conversation_memory import ConversationMemory

async def quick_query():
    # Get free tier config
    config = BillingLayer.generate_config("user@example.com")
    
    # Initialize components
    vector_db = VectorStoreManager(collection_name="public_tier_1")
    memory_graph = MemoryGraph()
    conversation_memory = ConversationMemory()
    
    # Ask a question
    result = await agi_brain(
        question="What is karma?",
        config=config,
        vector_db=vector_db,
        memory_graph=memory_graph,
        conversation_memory=conversation_memory,
        conversation_id="my-session"
    )
    
    print(result["final_answer"])

asyncio.run(quick_query())
```

### Example 2: Using Different Tiers

```python
# Free tier
config_free = BillingLayer.generate_config("free@example.com")

# Paid tier
config_paid = BillingLayer.generate_config("paid@example.com")

# Jarvis tier (founder only)
config_jarvis = BillingLayer.generate_config("chirag@example.com")

print(f"Free: {config_free['max_docs']} docs, {config_free['max_tokens']} tokens")
print(f"Paid: {config_paid['max_docs']} docs, {config_paid['max_tokens']} tokens")
print(f"Jarvis: Unlimited")
```

### Example 3: API Server

```bash
# Start the API server
python api/custom_api.py
```

```bash
# Make requests
curl -X POST http://localhost:8080/chat \
  -H "Authorization: Bearer dummy_token" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain dharma",
    "conversation_id": "user-123"
  }'
```

## Demo Modes

The demo script has multiple modes:

```bash
# Basic query (default: free tier)
python demo.py basic

# Basic query with specific tier
python demo.py basic jarvis

# Compare different tiers
python demo.py multi-tier

# Multi-turn conversation
python demo.py conversation

# Interactive mode
python demo.py interactive
```

## Architecture Overview

```
USER QUERY
    ↓
Layer 1: Intent Decomposition → Break into sub-goals
    ↓
Layer 2: Query Expansion → Generate multiple queries
    ↓
Layer 3: Source Routing → Choose Vector DB, Graph, etc.
    ↓
Layer 4: Memory Graph → THE BRAIN (understanding)
    ↓
Layer 5: Reasoning → Synthesize answer
    ↓
FINAL ANSWER
```

## Tier Differences

| Feature | Free | Paid | Jarvis |
|---------|------|------|--------|
| Max Docs | 5 | 20 | Unlimited |
| Deep Reasoning | No | Yes | Yes |
| Max Tokens | 2048 | 8192 | 32768 |
| Tools | Basic | Standard | All + Agency |
| Memory | Short | Medium | Long-term |
| Scriptures | Public | Public | Public + Private |

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip install torch
```

### "Cannot connect to Qdrant"

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker start <container-id>
```

### "Database connection failed"

```bash
# Check PostgreSQL
docker ps | grep postgres

# Update DATABASE_URL in .env
```

### "spaCy model not found"

```bash
python -m spacy download en_core_web_sm
```

## Next Steps

1. **Add Scripture PDFs**: Place PDF files in `data/scriptures/public/`
2. **Ingest Data**: Run the ingestion script to load PDFs into vector DB
3. **Customize Tiers**: Edit `billing/billing.py` to adjust tier configurations
4. **Add New Phases**: Extend Layer 4 with custom phases
5. **Deploy**: Use Docker or deploy to cloud

## Resources

- [Full Documentation](README.md)
- [Blueprint](Blueprint.md) - Complete architecture specification
- [Verification Guide](AGI%20VERIFICATION%20APPROACH.txt)

## Support

For issues or questions:
1. Check existing documentation
2. Review error logs in `logs/`
3. Open an issue on GitHub (when available)

## Philosophy

This AGI combines:
- **Ancient Wisdom**: Vedas, Gita, Upanishads
- **Modern AI**: Neural networks, embeddings
- **Hybrid Architecture**: Symbolic + neural

Creating an AI that truly **understands** rather than just processes text.

---

**"धर्मो रक्षति रक्षितः" - Dharma protects those who protect it**

Ready to explore Vedic intelligence? Start with `python demo.py interactive`
