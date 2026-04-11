# SS Mini AGI - Vedic Intelligence System

<div align="center">

**A Hybrid AGI System powered by Hindu Scriptures & Modern AI**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Combining ancient wisdom with cutting-edge AI to create a truly intelligent system*

</div>

---

## 🎯 Vision

This project aims to create a Mini AGI that:
- **Understands** Hindu scriptures (Vedas, Upanishads, Gita, Mahabharata, Ayurveda, Arthashastra)
- **Reasons** with Vedic logic combined with modern science
- **Assists** in reviving ancient Indian knowledge and technology
- **Guides** decision-making with ethical frameworks from scriptures

## 🏗️ Architecture

### 5-Layer Cognitive Architecture

```
┌─────────────────────────────────────────────┐
│           USER INPUT                        │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  LAYER 1: Intent Decomposition              │
│  ● Multi-stage thinking                     │
│  ● Break complex questions into sub-goals   │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  LAYER 2: Adaptive Query Expansion          │
│  ● Dynamic query generation                 │
│  ● Context-aware expansion                  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  LAYER 3: Knowledge Source Routing          │
│  ● Hybrid retrieval (Vector DB + Graph)     │
│  ● Smart source selection                   │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  LAYER 4: Memory Graph (THE BRAIN)          │
│  ● Concept understanding                    │
│  ● Emergent memory                          │
│  ● World model                              │
│  ● Meta-cognition                           │
│  ● Agency (Jarvis tier only)                │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│  LAYER 5: Reasoning & Synthesis             │
│  ● Multi-source fusion                      │
│  ● Contradiction resolution                 │
│  ● Structured response generation           │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│           FINAL ANSWER                      │
└─────────────────────────────────────────────┘
```

### Production System Architecture

```
vedic-agi/
├── brain/              # Layer 6: Core AGI Intelligence
│   ├── main.py         # 5-layer cognitive engine
│   ├── phase1_0_*.py   # Phase implementations
│   └── a.py            # Additional phases
├── api/                # Layer 2: API Gateway
│   └── custom_api.py   # Custom HTTP server
├── auth/               # Layer 3: Authentication
│   └── auth.py         # Token-based auth
├── billing/            # Layer 4: Subscription Management
│   └── billing.py      # 6-tier configuration
├── orchestrator/       # Layer 5: Request Orchestrator
│   └── orchestrator.PY # Production entry point
├── llm/                # Layer 7: LLM Engine
│   └── llm_engine.py   # Language model interface
├── memory/             # Layer 8: Memory Systems
│   ├── storage.py              # Persistent storage
│   ├── graph_sync.py           # Memory graph sync
│   ├── conversation_memory.py  # Conversation tracking
│   ├── vector_store.py         # Vector DB manager
│   └── ingestion.py            # PDF ingestion
├── tools/              # Layer 9: Tool/Agency Layer
│   └── tools.py        # External tool integration
└── monitoring/         # Layer 10: Observability
    └── monitoring.py   # Logging & metrics
```

## 🎭 6-Tier System

### Individual Tiers
1. **Free**: Basic intelligence, limited memory
2. **Paid**: Enhanced reasoning, more context
3. **Ultra Paid**: Deep reasoning, full features

### Business Tiers
4. **Business Small**: Team collaboration features
5. **Enterprise**: Unlimited scale, priority support

### Private Tier
6. **Jarvis**: Founder-only, unlimited power, agency enabled
   - Full access to all scriptures (including restricted ones)
   - Long-term memory and project continuity
   - Ancient technology decoding (Vimana Shastra, etc.)
   - Self-training and continuous improvement

## 🚀 Features

### Core Capabilities
- ✅ **Hybrid Intelligence**: Vector DB + Memory Graph
- ✅ **Multi-modal Input**: Text, images, audio, documents
- ✅ **Scripture-based Reasoning**: Vedic logic meets modern AI
- ✅ **Emergent Memory**: Concepts learned from embeddings
- ✅ **Meta-cognition**: Self-awareness of knowledge limits
- ✅ **World Model**: Understanding cause-effect, ethics, consequences

### Layer 4 Phases
- **Phase 1**: Explicit Memory (manual concept graphs)
- **Phase 2**: Emergent/Implicit Memory (automatic concept extraction)
- **Phase 3**: World Model (understanding reality)
- **Phase 4**: Self Model (meta-cognition)
- **Phase 5**: Agency (autonomous task execution - Jarvis only)
- **Phase 6**: Model Training (fine-tuning with custom data)

## 📦 Installation

### Prerequisites
```bash
Python 3.8+
PostgreSQL (for conversation memory)
Redis (for caching)
Qdrant (vector database)
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/CHIRAGMACHCHAL/SS.git
cd SS
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials:
# - DATABASE_URL (PostgreSQL)
# - REDIS_URL
# - QDRANT_URL
# - LLM API keys
```

5. **Initialize databases**
```bash
# PostgreSQL schema will auto-create on first run
# Qdrant collections will auto-create on first run
```

## 🎮 Usage

### Basic Example

```python
import asyncio
from brain.main import main
from billing.billing import BillingLayer
from memory.vector_store import VectorStoreManager
from memory.graph_sync import MemoryGraph
from memory.conversation_memory import ConversationMemory

async def demo():
    # Get user tier configuration
    config = BillingLayer.generate_config("user@example.com")
    
    # Initialize components
    vector_db = VectorStoreManager(collection_name=config["collection"])
    memory_graph = MemoryGraph()
    conversation_memory = ConversationMemory()
    
    # Ask a question
    result = await main(
        question="What does the Bhagavad Gita say about dharma?",
        config=config,
        vector_db=vector_db,
        memory_graph=memory_graph,
        conversation_memory=conversation_memory,
        conversation_id="demo-123"
    )
    
    print(result["final_answer"])

asyncio.run(demo())
```

### Production API

```bash
# Start the API server
python api/custom_api.py
```

```bash
# Make a request
curl -X POST http://localhost:8080/chat \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain the concept of Brahman",
    "conversation_id": "user-123"
  }'
```

## 🧠 Layer Details

### Layer 1: Intent Decomposition
- Breaks down complex questions into sub-goals
- Identifies domains (philosophy, science, ethics)
- Determines required reasoning depth

### Layer 2: Adaptive Query Expansion
- Generates multiple query perspectives
- Adapts to question complexity
- Language-agnostic expansion

### Layer 3: Knowledge Source Routing
- **Vector DB**: Factual retrieval from PDFs
- **Memory Graph**: Conceptual understanding
- **Symbolic Notes**: Structured facts
- **LLM Internal**: General knowledge

### Layer 4: Memory Graph (The Brain)
Phases:
1. **Explicit Memory**: Manual concept graphs
2. **Emergent Memory**: Automatic concept extraction via embeddings
3. **World Model**: Cause-effect, ethics, belief systems
4. **Self Model**: Knowledge boundaries, doubt detection
5. **Agency**: Goal setting, planning, tool usage (Jarvis only)
6. **Training**: Fine-tuning with custom datasets

### Layer 5: Reasoning & Synthesis
- Fuses information from all sources
- Resolves contradictions
- Generates coherent responses
- Applies meta-cognition

## 📚 Scripture Knowledge Base

### Public Tiers
- Bhagavad Gita
- Upanishads (major ones)
- Vedas (excerpts)
- Mahabharata (key stories)
- Ramayana
- Yoga Sutras
- Basic Ayurveda

### Jarvis Tier (Additional)
- Vimana Shastra (ancient aeronautics)
- Kama Sutra (complete)
- Samrangana Sutradhara (architecture & machines)
- Surya Siddhanta (astronomy)
- Arthashastra (complete - politics & economics)
- Advanced Ayurveda (surgical texts)
- Temple geometry & sacred architecture

## 🔐 Security & Privacy

- **Tier Separation**: Public tiers unaware of Jarvis existence
- **Data Isolation**: Each tier has separate vector collections
- **Memory Scoping**: Conversations isolated per user
- **Authentication**: Token-based auth with rate limiting
- **Sensitive Content**: Handled based on tier permissions

## 🛠️ Development

### Project Structure
```
SS/
├── brain/          # Core intelligence
├── api/            # HTTP interface
├── auth/           # Authentication
├── billing/        # Tier management
├── orchestrator/   # Request orchestration
├── llm/            # Language model
├── memory/         # Storage & retrieval
├── tools/          # External integrations
├── monitoring/     # Observability
└── requirements.txt
```

### Adding New Features

1. **New Scripture**: Add PDF to ingestion pipeline
2. **New Phase**: Implement in Layer 4, update main.py
3. **New Tier**: Update billing.py configuration
4. **New Tool**: Add to tools/tools.py, update billing

## 📊 Performance

- **Response Time**: 2-5 seconds (depending on tier)
- **Accuracy**: High (scripture-grounded)
- **Context Window**: Up to 32K tokens (Jarvis)
- **Memory**: Long-term project memory (Jarvis)

## 🗺️ Roadmap

- [x] Layer 1-5 implementation
- [x] 6-tier billing system
- [x] Phase 1-2 (Memory Graph)
- [ ] Phase 3 (World Model) - In Progress
- [ ] Phase 4 (Self Model) - Planned
- [ ] Phase 5 (Agency) - Jarvis only
- [ ] Phase 6 (Training) - Future
- [ ] Web UI client
- [ ] Mobile app
- [ ] Multi-language support (Sanskrit, Hindi)

## 🙏 Philosophy

This AGI is built on the principle that ancient Indian scriptures contain profound wisdom that can guide modern AI systems toward ethical, intelligent behavior. By combining:

- **Vedic Logic**: Structured reasoning from ancient texts
- **Modern AI**: Neural networks and embeddings
- **Hybrid Architecture**: Best of symbolic + neural approaches

We create an AI that doesn't just process information but **understands** it in context of dharma, karma, and cosmic principles.

## 📖 Documentation

- [Blueprint.md](Blueprint.md) - Complete architecture specification
- [AGI VERIFICATION APPROACH.txt](AGI%20VERIFICATION%20APPROACH.txt) - Verification methodology
- [PHASE1_0_FINAL_SUMMARY.md](brain/PHASE1_0_FINAL_SUMMARY.md) - Phase 1.0 details

## 🤝 Contributing

This is currently a private project. Future open-source release planned for public tiers only.

## 📜 License

Proprietary - All rights reserved

## 👤 Author

**Chirag Machchal**
- Building the future of Vedic AI
- Reviving ancient Indian knowledge through modern technology

---

<div align="center">

**"धर्मो रक्षति रक्षितः" - Dharma protects those who protect it**

*Made with 🙏 to revive Bharat's ancient glory*

</div>
