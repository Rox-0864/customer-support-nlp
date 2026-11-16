"""
business_rules.py — Taxonomía canónica y reglas de negocio post-pipeline.

Fuente única de verdad para:
  - Etiquetas de intención canónicas (español, alineadas con src/models/intent.py).
  - Intención sintética "delete_account", producida por reglas (no por el modelo).
  - Mapeo macro-intención y nombres legibles para la UI.
  - Overrides heurísticos de intención/sentimiento sobre la salida del modelo.
  - Categorización de riesgo de churn (Alto/Medio/Bajo) a partir del score.

Este módulo es Python puro (sin torch/transformers) y 100% testeable offline.
"""

from typing import Tuple

# ---------------------------------------------------------------------------
# Taxonomía de intenciones
# ---------------------------------------------------------------------------

# Etiquetas que predice el modelo (9 clases, español).
INTENT_LABELS = [
    "cancelacion", "consulta_general", "facturacion_pago", "feedback",
    "gestion_cuenta", "modificacion_pedido", "queja", "reembolso", "seguimiento",
]

# Intención sintética generada solo por reglas de negocio (no es clase del modelo).
DELETE_ACCOUNT_INTENT = "delete_account"

# Etiquetas de sentimiento (3 clases, en minúscula — coinciden con src/models/sentiment.py).
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------
# Mapas de presentación
# ---------------------------------------------------------------------------

MACRO_INTENT_MAP = {
    "cancelacion": "Gestión de Pedidos",
    "modificacion_pedido": "Gestión de Pedidos",
    "seguimiento": "Gestión de Pedidos",
    "reembolso": "Pagos y Reembolsos",
    "facturacion_pago": "Pagos y Reembolsos",
    "gestion_cuenta": "Gestión de Cuenta",
    DELETE_ACCOUNT_INTENT: "Gestión de Cuenta",
    "queja": "Quejas y Reclamos",
    "feedback": "Feedback",
    "consulta_general": "Atención al Cliente",
}

INTENT_DISPLAY = {
    "cancelacion": "Cancelación",
    "consulta_general": "Consulta General",
    "facturacion_pago": "Facturación/Pago",
    "feedback": "Feedback",
    "gestion_cuenta": "Gestión de Cuenta",
    "modificacion_pedido": "Modificación Pedido",
    "queja": "Queja",
    "reembolso": "Reembolso",
    "seguimiento": "Seguimiento",
    DELETE_ACCOUNT_INTENT: "Eliminación de Cuenta",
}

DEFAULT_MACRO_INTENT = "Otras Consultas"


def macro_intent(intent: str) -> str:
    """Agrupa una micro-intención en su macro-intención de negocio."""
    return MACRO_INTENT_MAP.get(intent, DEFAULT_MACRO_INTENT)


def display_intent(intent: str) -> str:
    """Nombre legible para UI de una micro-intención."""
    return INTENT_DISPLAY.get(intent, intent.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Keywords de las reglas de negocio
# ---------------------------------------------------------------------------

DELETE_ACCOUNT_KEYWORDS = [
    "excluir", "apagar minha conta", "cancelar conta",
    "borrar mi cuenta", "cancelar mi cuenta", "eliminar",
    "elimino mi cuenta", "cerrar mi cuenta", "dar de baja", "borrar cuenta",
]

CANCELACION_KEYWORDS = [
    "cancel", "anul", "baja", "baixa", "desuscrib", "desinscrev",
    "no me interesa seguir", "no quiero seguir", "quiero darme de baja",
]

SUSCRIPCION_KEYWORDS = [
    "suscrip", "assinatur", "subscription",
    "sócio", "socio", "plan", "plano",
    "membresía", "membresia", "servicio", "serviço",
    "contrato", "mensualidad", "cuota",
]

# Reglas de modificación de pedido: (grupo AND, grupo OR).
MODIFY_ORDER_RULES = [
    (("cambi", "modific"), ("talla", "color", "pedido")),
    (("pedí",), ("necesito",)),
]

QUEJA_KEYWORDS = [
    "pesim", "pésim", "pessim", "péssim", "mal servicio", "pésimo servicio",
    "pessimo atendimento", "no responden", "nadie responde", "pésima atención",
]

FEEDBACK_POSITIVO_KEYWORDS = [
    "gracias", "excelente", "muy rápido", "muy rapido", "eficiente", "genial",
    "perfecto", "buen servicio", "great", "excellent", "thank",
]

# Intenciones que no deberían convivir con un sentimiento positivo.
RISK_INTENTS_POSITIVE_OVERRIDE = {
    "cancelacion", "queja", "reembolso", "facturacion_pago",
}

# ---------------------------------------------------------------------------
# Reglas de negocio (overrides post-pipeline)
# ---------------------------------------------------------------------------


def _has_any(text: str, keywords) -> bool:
    return any(kw in text for kw in keywords)


def _es_cancelacion_suscripcion(text: str) -> bool:
    """Detecta intención de cancelar suscripción o servicio."""
    if not _has_any(text, CANCELACION_KEYWORDS):
        return False
    if _has_any(text, SUSCRIPCION_KEYWORDS):
        return True
    if "cancel" in text and _has_any(text, ["plan", "suscrip", "assinatur"]):
        return True
    if "baja" in text and _has_any(text, ["no me interesa", "no quiero", "quiero darme"]):
        return True
    return False


def apply_business_rules(text: str, intent: str, sentiment: str) -> Tuple[str, str]:
    """Aplica overrides heurísticos sobre la salida del modelo.

    Args:
        text: Texto crudo del mensaje.
        intent: Intención predicha (etiqueta española canónica).
        sentiment: Sentimiento predicho ("negative"/"neutral"/"positive").

    Returns:
        Tuple (intent, sentiment) corregidos.
    """
    texto = (text or "").lower()
    sentiment = (sentiment or "neutral").lower()

    # 1. Eliminación de cuenta → delete_account
    if _has_any(texto, DELETE_ACCOUNT_KEYWORDS) and ("cuenta" in texto or "conta" in texto):
        return DELETE_ACCOUNT_INTENT, sentiment

    # 2. Cancelación de suscripción → cancelacion
    if _es_cancelacion_suscripcion(texto):
        return "cancelacion", sentiment

    # 3. Modificación de pedido → modificacion_pedido (y suaviza sentimiento negativo)
    es_modificacion = False
    for and_group, or_group in MODIFY_ORDER_RULES:
        if _has_any(texto, and_group) and _has_any(texto, or_group):
            es_modificacion = True
            break
    if es_modificacion:
        intent = "modificacion_pedido"
        if sentiment == "negative":
            sentiment = "neutral"
        return intent, sentiment

    # 4. Sentimiento positivo incompatible con intención de riesgo → feedback
    if sentiment == "positive" and intent in RISK_INTENTS_POSITIVE_OVERRIDE:
        return "feedback", sentiment

    # 5. Intención de consulta con lenguaje de queja → queja
    if intent == "consulta_general" and _has_any(texto, QUEJA_KEYWORDS):
        sentiment = "negative" if sentiment != "negative" else sentiment
        return "queja", sentiment

    # 6. Consulta con feedback positivo → feedback
    if sentiment == "positive" and intent == "consulta_general" and _has_any(texto, FEEDBACK_POSITIVO_KEYWORDS):
        return "feedback", sentiment

    return intent, sentiment


# ---------------------------------------------------------------------------
# Categorización de riesgo de churn
# ---------------------------------------------------------------------------

CHURN_HIGH_THRESHOLD = 0.5
CHURN_MEDIUM_THRESHOLD = 0.25

RISK_HIGH = "Alto Riesgo"
RISK_MEDIUM = "Riesgo Medio"
RISK_LOW = "Riesgo Bajo"


def categorize_churn(score: float, intent: str = None) -> str:
    """Convierte el score de churn (0-1) en una categoría de riesgo de negocio.

    La eliminación de cuenta se considera siempre de alto riesgo,
    independientemente del score.
    """
    if intent == DELETE_ACCOUNT_INTENT:
        return RISK_HIGH
    if score >= CHURN_HIGH_THRESHOLD:
        return RISK_HIGH
    if score >= CHURN_MEDIUM_THRESHOLD:
        return RISK_MEDIUM
    return RISK_LOW
