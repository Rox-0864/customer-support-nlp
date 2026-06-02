import pytest
from scripts.pipeline_fallback import (
    FallbackPipeline,
    detect_language,
    SENTIMENT_KEYWORDS,
    INTENT_PATTERNS,
    FRUSTRATION_KEYWORDS,
    HIGH_CHURN_INTENTS,
)


@pytest.fixture
def pipeline():
    return FallbackPipeline()


class TestDetectLanguage:
    def test_detect_es(self):
        assert detect_language("hola buenos dias") == "es"

    def test_detect_pt(self):
        assert detect_language("não obrigado") == "pt"

    def test_detect_pt_with_multiple_markers(self):
        assert detect_language("péssimo atendimento, não funciona") == "pt"

    def test_detect_empty(self):
        assert detect_language("") == "es"


class TestSentiment:
    def test_positive_es(self, pipeline):
        result = pipeline.predict_sentiment("Excelente servicio, muchas gracias")
        assert result["label"] == "positive"
        assert result["probability"] > 0.3

    def test_negative_es(self, pipeline):
        result = pipeline.predict_sentiment("Pésimo servicio, terrible experiencia")
        assert result["label"] == "negative"
        assert result["probability"] > 0.3

    def test_neutral_es(self, pipeline):
        result = pipeline.predict_sentiment("Consulta sobre el horario de atención")
        assert result["label"] == "neutral"

    def test_positive_pt(self, pipeline):
        result = pipeline.predict_sentiment("Excelente atendimento, muito obrigado")
        assert result["label"] == "positive"

    def test_negative_pt(self, pipeline):
        result = pipeline.predict_sentiment("Péssimo, não funciona nada")
        assert result["label"] == "negative"

    def test_empty_text(self, pipeline):
        result = pipeline.predict_sentiment("")
        assert result["label"] == "neutral"
        assert result["probability"] == 1.0

    def test_negation_flips_sentiment(self, pipeline):
        result = pipeline.predict_sentiment("Pésimo servicio, no es bueno")
        assert result["label"] == "negative"

    def test_has_probabilities(self, pipeline):
        result = pipeline.predict_sentiment("Excelente")
        assert "probabilities" in result
        assert "negative" in result["probabilities"]
        assert "neutral" in result["probabilities"]
        assert "positive" in result["probabilities"]

    def test_sentiment_keywords_not_empty(self):
        assert len(SENTIMENT_KEYWORDS["positive"]["es"]) > 0
        assert len(SENTIMENT_KEYWORDS["negative"]["es"]) > 0
        assert len(SENTIMENT_KEYWORDS["neutral"]["es"]) > 0


class TestIntent:
    def test_cancelacion_es(self, pipeline):
        result = pipeline.predict_intent("Quiero la cancelación del pedido")
        assert result["intent"] == "cancelacion"

    def test_queja_es(self, pipeline):
        result = pipeline.predict_intent("Esto es inaceptable, pésimo servicio")
        assert result["intent"] == "queja"

    def test_consulta_general_es(self, pipeline):
        result = pipeline.predict_intent("Quiero saber el horario de atención")
        assert result["intent"] == "consulta_general"

    def test_feedback_es(self, pipeline):
        result = pipeline.predict_intent("Excelente servicio, me encantó")
        assert result["intent"] == "feedback"

    def test_reembolso_es(self, pipeline):
        result = pipeline.predict_intent("Quiero que me devuelvan mi dinero")
        assert result["intent"] == "reembolso"

    def test_facturacion_es(self, pipeline):
        result = pipeline.predict_intent("Tengo un problema con mi factura")
        assert result["intent"] == "facturacion_pago"

    def test_gestion_cuenta_es(self, pipeline):
        result = pipeline.predict_intent("Quiero cambiar mi contraseña")
        assert result["intent"] == "gestion_cuenta"

    def test_seguimiento_es(self, pipeline):
        result = pipeline.predict_intent("Cuál es el estado de mi envío?")
        assert result["intent"] == "seguimiento"

    def test_cancelacion_pt(self, pipeline):
        result = pipeline.predict_intent("Quero o cancelamento da assinatura")
        assert result["intent"] == "cancelacion"

    def test_queja_pt(self, pipeline):
        result = pipeline.predict_intent("você não presta, serviço péssimo")
        assert result["intent"] == "queja"

    def test_empty_text(self, pipeline):
        result = pipeline.predict_intent("")
        assert result["intent"] == "consulta_general"

    def test_probability_range(self, pipeline):
        result = pipeline.predict_intent("Quiero cancelar mi pedido urgentemente")
        assert 0.0 <= result["probability"] <= 1.0

    def test_intent_patterns_defined(self):
        expected_intents = {"cancelacion", "consulta_general", "facturacion_pago", "feedback",
                           "gestion_cuenta", "modificacion_pedido", "queja", "reembolso", "seguimiento"}
        assert set(INTENT_PATTERNS.keys()) == expected_intents


class TestChurn:
    def test_high_churn(self, pipeline):
        sentiment = pipeline.predict_sentiment("Pésimo, quiero cancelar todo")
        intent = pipeline.predict_intent("Quiero cancelar mi pedido")
        result = pipeline.compute_churn("Pésimo, quiero cancelar todo", sentiment, intent)
        assert result["aggregate_score"] > 0.3

    def test_low_churn(self, pipeline):
        sentiment = pipeline.predict_sentiment("Excelente servicio, gracias")
        intent = pipeline.predict_intent("Excelente servicio, gracias")
        result = pipeline.compute_churn("Excelente servicio, gracias", sentiment, intent)
        assert result["aggregate_score"] <= 0.5

    def test_high_churn_intents(self):
        assert "cancelacion" in HIGH_CHURN_INTENTS
        assert "queja" in HIGH_CHURN_INTENTS
        assert "reembolso" in HIGH_CHURN_INTENTS

    def test_frustration_keywords_not_empty(self):
        assert len(FRUSTRATION_KEYWORDS["es"]) > 0
        assert len(FRUSTRATION_KEYWORDS["pt"]) > 0


class TestPipeline:
    def test_predict_full_output(self, pipeline):
        result = pipeline.predict("Quiero cancelar mi pedido")
        assert "text" in result
        assert "sentiment" in result
        assert "intent" in result
        assert "churn" in result
        assert result["text"] == "Quiero cancelar mi pedido"

    def test_batch_predict(self, pipeline):
        texts = ["Excelente servicio", "Pésimo, quiero cancelar", "Consulta de horario"]
        results = pipeline.batch_predict(texts)
        assert len(results) == 3
        for r in results:
            assert "text" in r
            assert "sentiment" in r
            assert "intent" in r
            assert "churn" in r

    def test_get_model_status(self, pipeline):
        status = pipeline.get_model_status()
        assert status == {"sentiment": "fallback", "intent": "fallback"}


class TestModificacionPedido:
    def test_modify_order_es(self, pipeline):
        result = pipeline.predict_intent("quiero la modificacion de mi pedido")
        assert result["intent"] == "modificacion_pedido"

    def test_modify_order_pt(self, pipeline):
        result = pipeline.predict_intent("Quero modificar meu pedido")
        assert result["intent"] == "modificacion_pedido"
