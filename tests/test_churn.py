import pytest
from src.features.churn import ChurnScorer, FRUSTRATION_KEYWORDS, RISK_INTENTS


def test_default_weights():
    scorer = ChurnScorer()
    assert scorer.get_weights() == {"w1": 0.3, "w2": 0.35, "w3": 0.35}


def test_set_weights():
    scorer = ChurnScorer()
    scorer.set_weights(0.5, 0.3, 0.2)
    assert scorer.get_weights() == {"w1": 0.5, "w2": 0.3, "w3": 0.2}


def test_score_high_churn():
    scorer = ChurnScorer()
    sentiment = {
        "label": "negative",
        "probability": 0.95,
        "probabilities": {"negative": 0.95, "neutral": 0.03, "positive": 0.02},
    }
    intent = {
        "intent": "cancelacion",
        "probability": 0.9,
        "probabilities": {
            "cancelacion": 0.9, "queja": 0.04, "reembolso": 0.01,
            "feedback": 0.03, "consulta": 0.01,
        },
    }
    result = scorer.score(sentiment, intent, "Desisto, quiero cancelar mi cuenta")
    assert result["aggregate_score"] >= 0.7
    assert "sentiment_contribution" in result
    assert "frustration_contribution" in result
    assert "intent_contribution" in result


def test_score_low_churn():
    scorer = ChurnScorer()
    sentiment = {
        "label": "positive",
        "probability": 0.95,
        "probabilities": {"positive": 0.95, "neutral": 0.03, "negative": 0.02},
    }
    intent = {
        "intent": "feedback",
        "probability": 0.9,
        "probabilities": {
            "feedback": 0.9, "consulta": 0.04, "queja": 0.005,
            "cancelacion": 0.003, "reembolso": 0.002,
        },
    }
    result = scorer.score(sentiment, intent, "Excelente servicio")
    assert result["aggregate_score"] <= 0.2


def test_score_contains_all_keys():
    scorer = ChurnScorer()
    sentiment = {"label": "negative", "probability": 1.0, "probabilities": {"negative": 1.0, "neutral": 0.0, "positive": 0.0}}
    intent = {"intent": "cancelacion", "probability": 1.0, "probabilities": {"cancelacion": 1.0}}
    result = scorer.score(sentiment, intent, "test")
    required = {"aggregate_score", "sentiment_contribution", "frustration_contribution", "intent_contribution"}
    assert required.issubset(result.keys())


def test_conversation_worst_message_wins():
    scorer = ChurnScorer()
    low = {"aggregate_score": 0.1, "sentiment_contribution": 0.0, "frustration_contribution": 0.0, "intent_contribution": 0.0}
    high = {"aggregate_score": 0.9, "sentiment_contribution": 0.5, "frustration_contribution": 0.2, "intent_contribution": 0.2}
    result = scorer.score_conversation([low, high, low])
    assert result["aggregate_score"] == 0.9


def test_conversation_empty():
    scorer = ChurnScorer()
    result = scorer.score_conversation([])
    assert result["aggregate_score"] == 0.0


def test_frustration_keywords_es():
    scorer = ChurnScorer()
    sent = {"label": "negative", "probability": 0.5, "probabilities": {"negative": 0.5, "neutral": 0.3, "positive": 0.2}}
    intent = {"intent": "consulta_general", "probability": 0.5, "probabilities": {}}
    result = scorer.score(sent, intent, "Pésimo servicio, no funciona")
    assert result["frustration_contribution"] > 0


def test_frustration_keywords_pt():
    scorer = ChurnScorer()
    sent = {"label": "negative", "probability": 0.5, "probabilities": {"negative": 0.5, "neutral": 0.3, "positive": 0.2}}
    intent = {"intent": "consulta_general", "probability": 0.5, "probabilities": {}}
    result = scorer.score(sent, intent, "Péssimo atendimento, não funciona")
    assert result["frustration_contribution"] > 0


def test_no_frustration_false_positive():
    scorer = ChurnScorer()
    sent = {"label": "positive", "probability": 0.95, "probabilities": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}}
    intent = {"intent": "feedback", "probability": 0.9, "probabilities": {}}
    result = scorer.score(sent, intent, "Todo está perfecto, gracias")
    assert result["frustration_contribution"] == 0.0


@pytest.mark.parametrize("text,expected", [
    ("queja", 1.0),
    ("reclamo", 1.0),
    ("não serve", 1.0),
    ("felicidades", 0.0),
    ("gracias", 0.0),
])
def test_detect_frustration_various(text, expected):
    assert ChurnScorer._detect_frustration(text) == expected


def test_risk_intents_defined():
    assert "queja" in RISK_INTENTS
    assert "cancelacion" in RISK_INTENTS
    assert "reembolso" in RISK_INTENTS


def test_frustration_keywords_not_empty():
    assert len(FRUSTRATION_KEYWORDS) > 0


def test_score_with_python_float_edge_cases():
    scorer = ChurnScorer()
    sent = {"label": "negative", "probability": 1.0, "probabilities": {"negative": 1.0, "neutral": 0.0, "positive": 0.0}}
    intent = {"intent": "cancelacion", "probability": 1.0, "probabilities": {"cancelacion": 1.0, "queja": 1.0, "reembolso": 1.0}}
    result = scorer.score(sent, intent, "queja horrible pesimo")
    assert 0.0 <= result["aggregate_score"] <= 1.0


def test_score_no_negative_sentiment():
    scorer = ChurnScorer()
    sent = {"label": "positive", "probability": 0.9, "probabilities": {"negative": 0.0, "neutral": 0.1, "positive": 0.9}}
    intent = {"intent": "consulta_general", "probability": 0.9, "probabilities": {}}
    result = scorer.score(sent, intent, "consulta sobre mi pedido")
    assert result["aggregate_score"] < 0.5
