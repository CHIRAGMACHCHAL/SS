#!/usr/bin/env python3
"""
SS Mini AGI - Main Entry Point
Quick launcher for the Vedic Intelligence System
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set default environment if .env exists
from pathlib import Path
if Path('.env').exists():
    from dotenv import load_dotenv
    load_dotenv()


async def run_query(question: str, tier: str = "free"):
    """
    Run a single query and return the answer
    
    Args:
        question: The question to ask
        tier: Tier level (free, paid, ultra_paid, jarvis)
    
    Returns:
        Answer string
    """
    from brain.main import main as agi_brain
    from billing.billing import BillingLayer
    from memory.vector_store import VectorStoreManager
    from memory.graph_sync import MemoryGraph
    from memory.conversation_memory import ConversationMemory
    
    # Map tier to email
    email_map = {
        "free": "free@example.com",
        "paid": "paid@example.com",
        "ultra_paid": "ultra@example.com",
        "jarvis": "chirag@example.com"
    }
    
    email = email_map.get(tier, "free@example.com")
    config = BillingLayer.generate_config(email)
    
    # Initialize components
    vector_db = VectorStoreManager(collection_name=config.get("collection", "public_tier_1"))
    memory_graph = MemoryGraph()
    conversation_memory = ConversationMemory()
    
    # Run query
    result = await agi_brain(
        question=question,
        config=config,
        vector_db=vector_db,
        memory_graph=memory_graph,
        conversation_memory=conversation_memory,
        conversation_id=f"cli-{tier}"
    )
    
    return result.get("final_answer", "No answer generated")


async def interactive():
    """
    Interactive mode for continuous queries
    """
    from brain.main import main as agi_brain
    from billing.billing import BillingLayer
    from memory.vector_store import VectorStoreManager
    from memory.graph_sync import MemoryGraph
    from memory.conversation_memory import ConversationMemory
    import uuid
    
    print("\n" + "="*70)
    print("SS MINI AGI - INTERACTIVE MODE")
    print("="*70)
    print("\nCommands:")
    print("  help    - Show this help")
    print("  tier    - Change tier")
    print("  clear   - Clear conversation")
    print("  quit    - Exit")
    print("="*70)
    
    # Initial setup
    tier = "free"
    email_map = {
        "free": "free@example.com",
        "paid": "paid@example.com",
        "ultra_paid": "ultra@example.com",
        "jarvis": "chirag@example.com"
    }
    
    email = email_map[tier]
    config = BillingLayer.generate_config(email)
    conversation_id = str(uuid.uuid4())
    
    # Initialize components
    vector_db = VectorStoreManager(collection_name=config.get("collection", "public_tier_1"))
    memory_graph = MemoryGraph()
    conversation_memory = ConversationMemory()
    
    print(f"\n✓ Initialized ({tier.upper()} tier)")
    
    while True:
        try:
            question = input("\n❓ Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() == "quit":
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == "help":
                print("\nCommands:")
                print("  help    - Show this help")
                print("  tier    - Change tier")
                print("  clear   - Clear conversation")
                print("  quit    - Exit")
                continue
            
            if question.lower() == "tier":
                print("\nAvailable tiers: free, paid, ultra_paid, jarvis")
                new_tier = input("Select tier: ").strip().lower()
                if new_tier in email_map:
                    tier = new_tier
                    email = email_map[tier]
                    config = BillingLayer.generate_config(email)
                    vector_db = VectorStoreManager(collection_name=config.get("collection", "public_tier_1"))
                    print(f"✓ Switched to {tier.upper()} tier")
                else:
                    print("❌ Invalid tier")
                continue
            
            if question.lower() == "clear":
                conversation_id = str(uuid.uuid4())
                print("✓ Conversation cleared")
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
            import traceback
            traceback.print_exc()


def main():
    """
    Main entry point with argument parsing
    """
    if len(sys.argv) < 2:
        print("SS Mini AGI - Vedic Intelligence System")
        print("\nUsage:")
        print("  python run.py interactive")
        print("  python run.py query 'What is dharma?'")
        print("  python run.py query 'What is dharma?' --tier=jarvis")
        print("\nExamples:")
        print("  python run.py interactive")
        print("  python run.py query 'Explain karma and dharma'")
        return
    
    command = sys.argv[1]
    
    if command == "interactive":
        asyncio.run(interactive())
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("Error: Please provide a question")
            print("Usage: python run.py query 'Your question here'")
            return
        
        question = sys.argv[2]
        
        # Parse tier from arguments
        tier = "free"
        for arg in sys.argv[3:]:
            if arg.startswith("--tier="):
                tier = arg.split("=")[1]
        
        answer = asyncio.run(run_query(question, tier))
        print("\n" + "="*70)
        print("ANSWER")
        print("="*70)
        print(answer)
        print("="*70 + "\n")
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: interactive, query")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
