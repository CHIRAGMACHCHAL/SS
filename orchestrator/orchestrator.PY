#orchestrator/orchestrator.py Orchestrator Layer - Production entry point, ties everything together
import os
import time
from typing import Dict, Any
from brain.main import main as agi_brain  # Tere AGI brain call (Layer 6)

# =========================
# ENVIRONMENT VARIABLES
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

class Orchestrator:
    """
    Production Orchestrator Layer - Ties production layers together
    - Injects billing config
    - Attaches conversation memory (Layer 8)
    - Enables tools per tier
    - Calls AGI Brain
    - Adds production retry & logging (not in blueprint)
    """

    def __init__(self):
        self.log_level = LOG_LEVEL

    def log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{level}] {timestamp} - {message}")

    def run(
        self,
        question: str,
        config: Dict[str, Any],
        conversation_id: str,
        user_email: str
    ) -> Dict[str, Any]:
        self.log(f"Orchestrator: Starting for user {user_email} (tier: {config['tier']})")

        # 1. Memory attach (Layer 8 - real call when Layer 8 done)
        memory = self._get_memory(conversation_id)
        self.log(f"Memory attached (length: {len(memory)} chars)")

        # 2. Tools enable per config
        enabled_tools = config.get("allowed_tools", [])
        self.log(f"Enabled tools: {enabled_tools}")

        # 3. Prepare payload for AGI Brain
        brain_payload = {
            "question": question,
            "config": config,
            "memory": memory,
            "enabled_tools": enabled_tools
        }

        # 4. Call AGI Brain (main.py)
        try:
            start_time = time.time()
            brain_output = agi_brain(**brain_payload)
            elapsed = time.time() - start_time
            self.log(f"AGI Brain completed in {elapsed:.2f} seconds")
        except Exception as e:
            self.log(f"AGI Brain error: {str(e)}", "ERROR")
            return {"response": "System error. Please try again.", "success": False}

        # 5. Production retry (server issues ke liye, blueprint mein nahi)
        if not brain_output:
            self.log("Retrying AGI Brain call", "WARNING")
            brain_output = agi_brain(**brain_payload)  # One retry

        # 6. Update memory (Layer 8)
        self._update_memory(conversation_id, question, brain_output.get("final_answer", ""))

        return {
            "response": brain_output.get("final_answer", "No response"),
            "success": True,
            "processing_time": elapsed
        }

    def _get_memory(self, conversation_id: str) -> str:
        # Real Layer 8 call (placeholder abhi)
        return ""  # Replace with real when Layer 8 done

    def _update_memory(self, conversation_id: str, question: str, answer: str):
        # Real Layer 8 call (placeholder abhi)
        pass  # Replace with real when Layer 8 done