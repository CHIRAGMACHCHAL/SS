# llm/llm_engine.py - Layer 7: LLM Organ (Agnostic Engine)

import os
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG FROM .env / BILLING
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# =========================
# BASE LLM INTERFACE (Agnostic)
# =========================
class BaseLLMEngine:
    """
    Blueprint: Brain ko sirf interface pata hona chahiye
    Koi bhi provider ho, same invoke method
    """
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("Implement invoke() in child class")

# =========================
# TRANSFORMERS ENGINE (Current)
# =========================
class TransformersEngine(BaseLLMEngine):
    """
    Current local engine - tera original code
    """
    def __init__(self, model_name: str, temperature: float = 0.3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.temperature = temperature

    def invoke(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=self.temperature,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# LLM ENGINE REGISTRY (Agnostic Switch)
# =========================
class LLMEngineRegistry:
    """
    Future mein provider switch yahin se
    aaj transformers, kal vLLM, parso OpenAI
    """
    _registry = {}

    @classmethod
    def register(cls, name: str, engine_cls):
        cls._registry[name] = engine_cls

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Engine '{name}' not registered")
        return cls._registry[name](**kwargs)

# Register current engine
LLMEngineRegistry.register("transformers", TransformersEngine)

# Default instance (Layer 7 ka main object)
llm_organ = LLMEngineRegistry.create(
    "transformers",
    model_name=MODEL_NAME,
    temperature=TEMPERATURE
)

# =========================
# Public Interface for Brain/Orchestrator
# =========================
def generate(prompt: str) -> str:
    """Brain yeh function call karega"""
    return llm_organ.invoke(prompt)

def get_current_config() -> Dict[str, Any]:
    """Orchestrator ko config batane ke liye"""
    return {
        "provider": "transformers",
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_new_tokens": 512
    }