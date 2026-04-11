#!/usr/bin/env python3
"""
SS Mini AGI - Demo Script
Demonstrates the core functionality of the Vedic Intelligence System
"""

import asyncio
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from brain.main import main as agi_brain
from billing.billing import BillingLayer
from memory.vector_store import VectorStoreManager
from memory.graph_sync import MemoryGraph
from memory.conversation_memory import ConversationMemory


async def demo_basic_query(tier="free"):
    """
    Demonstrates a basic query to the AGI system
    """
    print("\n" + "="*70)
    print(f"DEMO: Basic Query ({tier.upper()} tier)")
    print("="*70)
    
    # Get configuration for the specified tier
    email = "demo@example.com" if tier != "jarvis" else "chirag@example.com"
    config = BillingLayer.generate_config(email)
    
    print(f"\n📋 Configuration:")
    print(f"   Tier: {config['tier']}")
    print(f"   Max Docs: {config['max_docs']}")
    print(f"   Deep Reasoning: {config['deep_reasoning']}")
    print(f"   Max Tokens: {config['max_tokens']}")
    
    # Initialize components
    collection = config.get("collection", "public_tier_1")
    vector_db = VectorStoreManager(collection_name=collection)
    memory_graph = MemoryGraph()
    conversation_memory = ConversationMemory()
    
    # Demo question
    question = "What is dharma according to Hindu philosophy?"
    
    print(f"\n❓ Question: {question}")
    print(f"\n🧠 Processing...")
    
    try:
        # Call the AGI brain
        result = await agi_brain(
            question=question,
            config=config,
            vector_db=vector_db,
            memory_graph=memory_graph,
            conversation_memory=conversation_memory,
            conversation_id=f"demo-{tier}-001"
        )
        
        print(f"\n✅ Answer:")
        print("-" * 70)
        print(result.get("final_answer", "No answer generated"))
        print("-" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


async def demo_multi_tier():
    """
    Demonstrates the difference between tiers
    """
    print("\n" + "="*70)
    print("DEMO: Multi-Tier Comparison")
    print("="*70)
    
    question = "Explain the relationship between karma and dharma"
    
    for tier_email in [
        ("free", "free@example.com"),
        ("paid", "paid@example.com"),
        ("jarvis", "chirag@example.com")
    ]:
        tier_name, email = tier_email
        
        print(f"\n\n{'─'*70}")
        print(f"🎭 TIER: {tier_name.upper()}")
        print(f"{'─'*70}")
        
        config = BillingLayer.generate_config(email)
        
        print(f"Configuration: max_docs={config['max_docs']}, "
              f"deep_reasoning={config['deep_reasoning']}, "
              f"max_tokens={config['max_tokens']}")
        
        # Note: In a real demo, you would call the AGI here
        # For this demo, we just show the config differences
        print(f"Allowed tools: {', '.join(config.get('allowed_tools', [])[:3])}...")
        
        if tier_name == "jarvis":
            print(f"⭐ Special capabilities: Agency, Ancient Tech, Unlimited Memory")


async def demo_conversation_flow():
    """
    Demonstrates a multi-turn conversation
    """
    print("\n" + "="*70)
    print("DEMO: Conversation Flow")
    print("="*70)
    
    config = BillingLayer.generate_config("demo@example.com")
    vector_db = VectorStoreManager(collection_name="public_tier_1")
    memory_graph = MemoryGraph()
    conversation_memory = ConversationMemory()
    conversation_id = "demo-conversation-001"
    
    questions = [
        "What are the four main goals of life in Hinduism?",
        "Tell me more about the first one",
        "How does this relate to modern life?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n\n{'─'*70}")
        print(f"Turn {i}: {question}")
        print(f"{'─'*70}")
        
        try:
            result = await agi_brain(
                question=question,
                config=config,
                vector_db=vector_db,
                memory_graph=memory_graph,
                conversation_memory=conversation_memory,
                conversation_id=conversation_id
            )
            
            answer = result.get("final_answer", "No answer")
            # Truncate for demo
            if len(answer) > 200:
                answer = answer[:200] + "..."
            
            print(f"Answer: {answer}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            break


def print_system_info():
    """
    Prints system information and architecture overview
    """
    print("\n" + "="*70)
    print("SS MINI AGI - VEDIC INTELLIGENCE SYSTEM")
    print("="*70)
    
    print("\n🏗️  5-Layer Cognitive Architecture:")
    print("   Layer 1: Intent Decomposition (Multi-stage thinking)")
    print("   Layer 2: Adaptive Query Expansion")
    print("   Layer 3: Knowledge Source Routing (Hybrid)")
    print("   Layer 4: Memory Graph (THE BRAIN)")
    print("   Layer 5: Reasoning & Synthesis")
    
    print("\n🎭 6-Tier System:")
    print("   Individual: Free, Paid, Ultra Paid")
    print("   Business: Business Small, Enterprise")
    print("   Private: Jarvis (Founder only)")
    
    print("\n📚 Knowledge Base:")
    print("   Public: Gita, Upanishads, Vedas, Mahabharata")
    print("   Jarvis: + Vimana Shastra, Arthashastra, Ancient Tech")
    
    print("\n🧠 Layer 4 Phases:")
    print("   Phase 1: Explicit Memory")
    print("   Phase 2: Emergent/Implicit Memory")
    print("   Phase 3: World Model")
    print("   Phase 4: Self Model (Meta-cognition)")
    print("   Phase 5: Agency (Jarvis only)")
    print("   Phase 6: Training (Future)")


async def interactive_mode():
    """
    Interactive demo mode
    """
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter questions or 'quit' to exit")
    
    # Setup
    tier_input = input("\nSelect tier (free/paid/jarvis) [free]: ").strip().lower()
    if tier_input not in ["free", "paid", "jarvis"]:
        tier_input = "free"
    
    email = "chirag@example.com" if tier_input == "jarvis" else f"{tier_input}@example.com"
    config = BillingLayer.generate_config(email)
    
    vector_db = VectorStoreManager(collection_name=config.get("collection", "public_tier_1"))
    memory_graph = MemoryGraph()
    conversation_memory = ConversationMemory()
    conversation_id = f"interactive-{tier_input}"
    
    print(f"\n✅ Initialized {tier_input.upper()} tier")
    print("─" * 70)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🧠 Processing...\n")
            
            result = await agi_brain(
                question=question,
                config=config,
                vector_db=vector_db,
                memory_graph=memory_graph,
                conversation_memory=conversation_memory,
                conversation_id=conversation_id
            )
            
            print("✅ Answer:")
            print("-" * 70)
            print(result.get("final_answer", "No answer generated"))
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


async def main():
    """
    Main demo function
    """
    print_system_info()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "basic":
            tier = sys.argv[2] if len(sys.argv) > 2 else "free"
            await demo_basic_query(tier)
        
        elif mode == "multi-tier":
            await demo_multi_tier()
        
        elif mode == "conversation":
            await demo_conversation_flow()
        
        elif mode == "interactive":
            await interactive_mode()
        
        else:
            print(f"\n❌ Unknown mode: {mode}")
            print_usage()
    
    else:
        print_usage()


def print_usage():
    """
    Prints usage information
    """
    print("\n" + "="*70)
    print("USAGE")
    print("="*70)
    print("\nRun demos:")
    print("  python demo.py basic [free|paid|jarvis]")
    print("  python demo.py multi-tier")
    print("  python demo.py conversation")
    print("  python demo.py interactive")
    print("\nExamples:")
    print("  python demo.py basic free")
    print("  python demo.py basic jarvis")
    print("  python demo.py interactive")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
