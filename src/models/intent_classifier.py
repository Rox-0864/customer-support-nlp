import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
from typing import Optional
import os
import json

@dataclass
class IntentResult:
    intent: str
    confidence: float
    lang: str
    model: str

MODEL_PATH = "data/models/intent_xlmr"
HF_REPO = "Rosela/xlmr-intent-6clases"
FALLBACK_LABELS = [
    "reclamo", "consulta", "cancelacion",
    "problema_tecnico", "facturacion", "resuelto"
]
_intent_pipeline = None
_pipeline_source = None

def _get_intent_pipeline():
    global _intent_pipeline, _pipeline_source
    if _intent_pipeline is not None:
        return _intent_pipeline

    # 1. Try local model first
    if os.path.exists(MODEL_PATH) and os.path.exists(f"{MODEL_PATH}/config.json"):
        has_safetensors = os.path.exists(f"{MODEL_PATH}/model.safetensors")
        if has_safetensors:
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
                _intent_pipeline = pipeline(
                    "text-classification", model=model, tokenizer=tokenizer, device=-1
                )
                _pipeline_source = "local"
                return _intent_pipeline
            except Exception:
                pass

    # 2. Try HuggingFace repo directly
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(HF_REPO)
        _intent_pipeline = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, device=-1
        )
        _pipeline_source = "hf"
        return _intent_pipeline
    except Exception:
        pass

    # 3. Fallback: zero-shot
    _intent_pipeline = pipeline(
        "zero-shot-classification",
        model="joeddav/xlm-roberta-large-xnli",
        device=-1
    )
    _pipeline_source = "zero-shot"
    return _intent_pipeline

def classify_intent(text: str, lang: Optional[str] = None) -> IntentResult:
    from src.utils.lang_detect import detect_lang

    if lang is None:
        lang = detect_lang(text)
    if lang not in ("es", "pt"):
        lang = "es"

    pipe = _get_intent_pipeline()

    if _pipeline_source in ("local", "hf"):
        result = pipe(text)[0]
        intent = result["label"]
        confidence = float(result["score"])
        model_name = "XLM-R Intent" if _pipeline_source == "local" else "XLM-R HF"
    else:
        result = pipe(text, FALLBACK_LABELS)
        intent = result["labels"][0]
        confidence = float(result["scores"][0])
        model_name = "XLM-R Zero-Shot"

    return IntentResult(intent=intent, confidence=confidence, lang=lang, model=model_name)

def batch_classify_intent(texts: list, lang: Optional[str] = None) -> list:
    from src.utils.lang_detect import detect_lang
    if lang is None:
        lang = detect_lang(texts[0]) if texts else "es"
    return [classify_intent(t, lang) for t in texts]
