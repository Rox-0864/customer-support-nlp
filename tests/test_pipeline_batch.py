import pytest
from src.pipeline import ConversaAIPipeline
from src.features.churn import ChurnScorer


class FakeSentiment:
    def __init__(self, calls):
        self.calls = calls

    def predict_sentiment_batch(self, texts):
        self.calls.append(list(texts))
        return [
            {
                "label": "neutral",
                "probability": 1.0,
                "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
            }
            for _ in texts
        ]


class FakeIntent:
    LABELS = [
        "cancelacion", "consulta_general", "facturacion_pago", "feedback",
        "gestion_cuenta", "modificacion_pedido", "queja", "reembolso", "seguimiento",
    ]

    def __init__(self, calls):
        self.calls = calls

    def predict_intent_batch(self, texts):
        self.calls.append(list(texts))
        probs = {lbl: 0.0 for lbl in self.LABELS}
        probs["consulta_general"] = 1.0
        return [
            {"intent": "consulta_general", "probability": 1.0, "probabilities": dict(probs)}
            for _ in texts
        ]


def _make_pipeline(sentiment_calls, intent_calls):
    p = object.__new__(ConversaAIPipeline)
    p.churn_scorer = ChurnScorer()
    p.sentiment_classifier = FakeSentiment(sentiment_calls)
    p.intent_classifier = FakeIntent(intent_calls)
    p.get_model_status = lambda: {"sentiment": "xlm-r", "intent": "xlm-r"}
    return p


def test_batch_predict_chunks_real_batches():
    sentiment_calls, intent_calls = [], []
    p = _make_pipeline(sentiment_calls, intent_calls)

    texts = ["a", "b", "c", "d", "e"]
    results = p.batch_predict(texts, batch_size=2)

    assert [r["text"] for r in results] == texts
    assert len(results) == 5

    # batch_size=2 sobre 5 no-vacíos -> 3 llamadas en batch real
    assert sentiment_calls == [["a", "b"], ["c", "d"], ["e"]]
    assert intent_calls == [["a", "b"], ["c", "d"], ["e"]]

    for r in results:
        assert "sentiment" in r
        assert "intent" in r
        assert "churn" in r
        assert "model_status" in r


def test_batch_predict_skips_empty_texts():
    sentiment_calls, intent_calls = [], []
    p = _make_pipeline(sentiment_calls, intent_calls)

    texts = ["", "hola", "   ", "chau"]
    results = p.batch_predict(texts, batch_size=16)

    # Los vacíos no llegan al clasificador en batch
    assert sentiment_calls == [["hola", "chau"]]
    assert intent_calls == [["hola", "chau"]]

    # Los vacíos devuelven resultado por defecto (neutral / consulta_general)
    assert results[0]["sentiment"]["label"] == "neutral"
    assert results[0]["intent"]["intent"] == "consulta_general"
    assert results[2]["sentiment"]["label"] == "neutral"
    assert results[2]["intent"]["intent"] == "consulta_general"

    # Orden preservado
    assert [r["text"] for r in results] == texts


def test_batch_predict_empty_list():
    sentiment_calls, intent_calls = [], []
    p = _make_pipeline(sentiment_calls, intent_calls)
    assert p.batch_predict([], batch_size=16) == []
