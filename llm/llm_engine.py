# llm/llm_engine.py - Layer 7: LLM Organ (Final - Tera detailed code + Production ready)

import os
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG FROM .env / BILLING (Production ready)
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# =========================
# BASE LLM INTERFACE (Agnostic) - Tera code
# =========================
class BaseLLMEngine:
    """
    Blueprint rule:
    - Brain never knows WHICH engine
    - Brain only knows WHAT it can do
    """
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLM engine must implement invoke()")

# =========================
# TRANSFORMERS ENGINE - Tera detailed code
# =========================
class TransformersEngine(BaseLLMEngine):
    """
    LangChain-free, Ollama-free LLM runtime
    Output: plain string (same as before)
    """
    def __init__(self, model_name: str, temperature: float = 0.3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=self.temperature,
            do_sample=True
        )
        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
    # 🔒 compatibility for training engine
    def invoke(self, prompt: str) -> str:
        return self.generate(prompt)

# =========================
# LLM ENGINE REGISTRY - Tera code
# =========================
class LLMEngineRegistry:
    """
    Central authority for engine selection
    Brain NEVER instantiates engines directly
    """
    _registry = {}

    @classmethod
    def register(cls, name: str, engine_cls):
        cls._registry[name] = engine_cls

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"LLM Engine '{name}' not registered")
        return cls._registry[name](**kwargs)

# Register current engine
LLMEngineRegistry.register("transformers", TransformersEngine)

# Default instance

llm = LLMEngineRegistry.create("transformers", model_name=MODEL_NAME, temperature=TEMPERATURE)
# =========================
# Public Interface for Brain/Orchestrator (Production ready)
# =========================
def generate(prompt: str) -> str:
    """Brain ya Orchestrator isse call karega"""
    return llm.invoke(prompt)

def get_current_config() -> Dict[str, Any]:
    """Orchestrator/Billing ke liye config"""
    return {
        "provider": "transformers",
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_new_tokens": 512
    }