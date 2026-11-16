import pytest
from src.features.business_rules import (
    apply_business_rules,
    categorize_churn,
    macro_intent,
    display_intent,
    INTENT_LABELS,
    DELETE_ACCOUNT_INTENT,
    RISK_HIGH,
    RISK_MEDIUM,
    RISK_LOW,
)


class TestApplyBusinessRules:
    def test_delete_account_es(self):
        intent, sentiment = apply_business_rules(
            "Quiero borrar mi cuenta", "consulta_general", "neutral"
        )
        assert intent == DELETE_ACCOUNT_INTENT

    def test_delete_account_pt(self):
        intent, sentiment = apply_business_rules(
            "Quero apagar minha conta", "consulta_general", "neutral"
        )
        assert intent == DELETE_ACCOUNT_INTENT

    def test_delete_account_requires_cuenta(self):
        intent, _ = apply_business_rules(
            "Quiero eliminar un archivo", "consulta_general", "neutral"
        )
        assert intent == "consulta_general"

    def test_modify_order_es(self):
        intent, sentiment = apply_business_rules(
            "Quiero cambiar la talla de mi pedido", "consulta_general", "negative"
        )
        assert intent == "modificacion_pedido"
        assert sentiment == "neutral"

    def test_modify_order_pedi_necesito(self):
        intent, _ = apply_business_rules(
            "Pedí algo y necesito cambiarlo", "seguimiento", "neutral"
        )
        assert intent == "modificacion_pedido"

    def test_cancelacion_suscripcion(self):
        intent, _ = apply_business_rules(
            "Quiero cancelar mi suscripción", "consulta_general", "neutral"
        )
        assert intent == "cancelacion"

    def test_positive_risk_intent_to_feedback(self):
        intent, _ = apply_business_rules(
            "Me encanta el servicio", "queja", "positive"
        )
        assert intent == "feedback"

    def test_queja_override(self):
        intent, sentiment = apply_business_rules(
            "Pésimo servicio, nadie responde", "consulta_general", "neutral"
        )
        assert intent == "queja"
        assert sentiment == "negative"

    def test_feedback_positivo(self):
        intent, _ = apply_business_rules(
            "Gracias, excelente atención", "consulta_general", "positive"
        )
        assert intent == "feedback"

    def test_no_override_passthrough(self):
        intent, sentiment = apply_business_rules(
            "Quiero saber el horario", "consulta_general", "neutral"
        )
        assert intent == "consulta_general"
        assert sentiment == "neutral"


class TestCategorizeChurn:
    def test_high(self):
        assert categorize_churn(0.9) == RISK_HIGH

    def test_medium(self):
        assert categorize_churn(0.3) == RISK_MEDIUM

    def test_low(self):
        assert categorize_churn(0.1) == RISK_LOW

    def test_delete_account_always_high(self):
        assert categorize_churn(0.0, DELETE_ACCOUNT_INTENT) == RISK_HIGH

    def test_boundary_high(self):
        assert categorize_churn(0.5) == RISK_HIGH

    def test_boundary_medium(self):
        assert categorize_churn(0.25) == RISK_MEDIUM


class TestMaps:
    def test_intent_labels_count(self):
        assert len(INTENT_LABELS) == 9

    def test_macro_intent_known(self):
        assert macro_intent("reembolso") == "Pagos y Reembolsos"

    def test_macro_intent_unknown(self):
        assert macro_intent("otra_cosa") == "Otras Consultas"

    def test_macro_intent_delete_account(self):
        assert macro_intent(DELETE_ACCOUNT_INTENT) == "Gestión de Cuenta"

    def test_display_intent(self):
        assert display_intent("facturacion_pago") == "Facturación/Pago"

    def test_display_intent_unknown(self):
        assert display_intent("algo_nuevo") == "Algo Nuevo"
