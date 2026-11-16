from __future__ import annotations
import logging
from collections import Counter
from typing import List

logger = logging.getLogger(__name__)

from src.data.processor import DataProcessor
from src.features.churn import ChurnScorer
from src.models.sentiment import SentimentClassifier
from src.models.intent import IntentClassifier


class ConversaAIPipeline:
    def __init__(self, sentiment_model: str = None,
                 intent_model: str = None,
                 tokenizer_model: str = None):
        sentiment_model = sentiment_model or "Rosela/xlm-r-sentiment-espt"
        intent_model = intent_model or "Rosela/xlm-r-intent-espt"
        tokenizer_model = tokenizer_model or "Rosela/xlm-r-sentiment-espt"
        self.data_processor = DataProcessor(model_name=tokenizer_model)
        self.sentiment_classifier = SentimentClassifier(self.data_processor, model_name=sentiment_model)
        self.intent_classifier = IntentClassifier(self.data_processor, model_name=intent_model)
        self.churn_scorer = ChurnScorer()

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return self._empty_result(text)
        sentiment = self.sentiment_classifier.predict_sentiment(text)
        intent = self.intent_classifier.predict_intent(text)
        churn = self.churn_scorer.score(sentiment, intent, text)

        return {
            "text": text,
            "sentiment": sentiment,
            "intent": intent,
            "churn": churn,
            "model_status": self.get_model_status(),
        }

    def batch_predict(self, texts: List[str], batch_size: int = 16) -> List[dict]:
        results: List[dict] = [None] * len(texts)
        nonempty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        status = self.get_model_status()

        for start in range(0, len(nonempty), batch_size):
            chunk = nonempty[start:start + batch_size]
            chunk_texts = [t for _, t in chunk]
            sentiments = self.sentiment_classifier.predict_sentiment_batch(chunk_texts)
            intents = self.intent_classifier.predict_intent_batch(chunk_texts)
            for (idx, text), sentiment, intent in zip(chunk, sentiments, intents):
                churn = self.churn_scorer.score(sentiment, intent, text)
                results[idx] = {
                    "text": text,
                    "sentiment": sentiment,
                    "intent": intent,
                    "churn": churn,
                    "model_status": status,
                }

        for i, text in enumerate(texts):
            if results[i] is None:
                results[i] = self._empty_result(text)
        return results

    def _empty_result(self, text: str) -> dict:
        intent_probs = {
            lbl: (1.0 if lbl == "consulta_general" else 0.0)
            for lbl in IntentClassifier.LABELS
        }
        return {
            "text": text,
            "sentiment": {
                "label": "neutral",
                "probability": 1.0,
                "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            },
            "intent": {
                "intent": "consulta_general",
                "probability": 1.0,
                "probabilities": intent_probs,
            },
            "churn": {
                "aggregate_score": 0.0,
                "sentiment_contribution": 0.0,
                "frustration_contribution": 0.0,
                "intent_contribution": 0.0,
            },
            "model_status": self.get_model_status(),
        }

    def predict_conversation(self, messages: List[str]) -> dict:
        processed = [self.predict(msg) for msg in messages]

        message_results = [
            {
                "text": p["text"],
                "sentiment": p["sentiment"],
                "intent": p["intent"],
                "churn": p["churn"],
            }
            for p in processed
        ]

        churn_scores = [p["churn"] for p in processed]
        aggregate_churn = self.churn_scorer.score_conversation(churn_scores)

        sentiments = [p["sentiment"]["label"] for p in processed]
        intents = [p["intent"]["intent"] for p in processed]

        dominant_sentiment = (
            Counter(sentiments).most_common(1)[0][0] if sentiments else "neutral"
        )
        dominant_intent = (
            Counter(intents).most_common(1)[0][0] if intents else "consulta_general"
        )

        return {
            "messages": message_results,
            "aggregate_churn": aggregate_churn,
            "dominant_sentiment": dominant_sentiment,
            "dominant_intent": dominant_intent,
        }

    def get_model_status(self) -> dict:
        return {
            "sentiment": (
                "xlm-r"
                if self.sentiment_classifier.is_model_loaded()
                else "fallback"
            ),
            "intent": (
                "xlm-r"
                if self.intent_classifier.is_model_loaded()
                else "fallback"
            ),
        }

