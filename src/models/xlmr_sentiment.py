from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class XlmrSentimentResult:
    label: str
    confidence: float
    lang: str

MODEL_PATH = "data/models/xlmr-sentiment"
_xlmr_pipeline = None

def _get_pipeline():
    global _xlmr_pipeline
    if _xlmr_pipeline is not None:
        return _xlmr_pipeline

    if os.path.exists(MODEL_PATH) and os.path.exists(f"{MODEL_PATH}/config.json"):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _xlmr_pipeline = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer, device=-1
        )
    else:
        _xlmr_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=-1
        )

    return _xlmr_pipeline

def classify(text: str, lang: Optional[str] = None) -> XlmrSentimentResult:
    from src.utils.lang_detect import detect_lang

    if lang is None:
        lang = detect_lang(text)
    if lang not in ("es", "pt"):
        lang = "es"

    pipe = _get_pipeline()

    if os.path.exists(MODEL_PATH) and os.path.exists(f"{MODEL_PATH}/config.json"):
        result = pipe(text)[0]
        label_raw = result["label"]
        confidence = float(result["score"])
        if "LABEL" in label_raw:
            idx = int(label_raw.split("_")[1])
            label = "positive" if idx == 1 else "negative"
        else:
            label = label_raw.lower()
    else:
        result = pipe(text)[0]
        label_str = result["label"]
        if "LABEL" in label_str:
            star = int(label_str.split("_")[1])
            if star <= 2:
                label = "negative"
            elif star == 3:
                label = "neutral"
            else:
                label = "positive"
        else:
            label = label_str.lower()
        confidence = float(result["score"])

    return XlmrSentimentResult(label=label, confidence=confidence, lang=lang)
