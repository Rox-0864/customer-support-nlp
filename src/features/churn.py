#!/usr/bin/env python3
from __future__ import annotations
"""
ChurnScorer — Score heurístico de riesgo de abandono (churn).

Combina señales de:
  - Sentimiento negativo (probabilidad de clase "negative")
  - Frustración detectada por keywords ES/PT
  - Intención de riesgo (queja, cancelacion, reembolso)

Fórmula:
  churn = w1 * sent_prob_negative + w2 * frust_flag + w3 * max(intent_risk_probs)

Pesos default: w1=0.3, w2=0.35, w3=0.35
"""


# ---------------------------------------------------------------------------
# Keywords de frustración — combinación ES + PT
# ---------------------------------------------------------------------------
FRUSTRATION_KEYWORDS = frozenset({
    # Español
    "desisto", "nunca más", "pésimo", "terrible", "horrible",
    "inaceptable", "pésimo servicio", "malísimo", "deficiente",
    "no sirve", "no funciona", "queja", "reclamo", "protesta",
    "abandono", "cancelación", "procon",
    # Portugués
    "nunca mais", "péssimo", "terrível", "horrível", "inaceitável",
    "péssimo serviço", "péssimo atendimento", "não serve",
    "não funciona", "reclamação", "reclamo", "protesto",
    "abandono", "cancelamento", "procon",
})

# Intents consideradas de alto riesgo de churn
RISK_INTENTS = frozenset({"queja", "cancelacion", "reembolso"})


class ChurnScorer:
    """Evalúa riesgo de churn combinando sentimiento, frustración e intención."""

    def __init__(self, w1: float = 0.3, w2: float = 0.35, w3: float = 0.35):
        """
        Args:
            w1: peso para contribución de sentimiento negativo
            w2: peso para contribución de frustración (keywords)
            w3: peso para contribución de intención de riesgo
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    # -- Pesos configurables ------------------------------------------------

    def get_weights(self) -> dict:
        """Retorna pesos actuales."""
        return {"w1": self.w1, "w2": self.w2, "w3": self.w3}

    def set_weights(self, w1: float, w2: float, w3: float) -> None:
        """Actualiza pesos en runtime."""
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    # -- Detección de frustración por keywords ------------------------------

    @staticmethod
    def _detect_frustration(text: str) -> float:
        """Retorna 1.0 si el texto contiene keywords de frustración, 0.0 si no.

        La comparación es case-insensitive y busca la keyword como substring
        (para capturar variaciones como "queja" dentro de "quejé", etc.).
        """
        text_lower = text.lower()
        for kw in FRUSTRATION_KEYWORDS:
            if kw in text_lower:
                return 1.0
        return 0.0

    # -- Score individual ---------------------------------------------------

    def score(self, sentiment: dict, intent: dict, text: str) -> dict:
        """Calcula churn score para un mensaje individual.

        Args:
            sentiment: {"label": str, "probability": float,
                        "probabilities": {str: float}}
            intent:    {"intent": str, "probability": float,
                        "probabilities": {str: float}}
            text:      Texto crudo del mensaje.

        Returns:
            dict con:
              - aggregate_score: float (churn compuesto en [0, 1])
              - sentiment_contribution: float
              - frustration_contribution: float
              - intent_contribution: float
        """
        # --- Sentimiento negativo ---
        sent_prob_negative = sentiment.get("probabilities", {}).get("negative", 0.0)
        sentiment_contribution = self.w1 * sent_prob_negative

        # --- Frustración por keywords ---
        frust_flag = self._detect_frustration(text)
        frustration_contribution = self.w2 * frust_flag

        # --- Intención de riesgo ---
        intent_probs = intent.get("probabilities", {})
        risk_values = [
            intent_probs.get(intent_name, 0.0)
            for intent_name in RISK_INTENTS
        ]
        max_risk_prob = max(risk_values) if risk_values else 0.0
        intent_contribution = self.w3 * max_risk_prob

        # --- Agregado ---
        aggregate_score = sentiment_contribution + frustration_contribution + intent_contribution

        # Acotar a [0, 1] por seguridad (puede exceder si pesos suman > 1)
        aggregate_score = min(max(aggregate_score, 0.0), 1.0)

        return {
            "aggregate_score": aggregate_score,
            "sentiment_contribution": sentiment_contribution,
            "frustration_contribution": frustration_contribution,
            "intent_contribution": intent_contribution,
        }

    # -- Score multi-mensaje (conversación) ---------------------------------

    def score_conversation(self, messages: list[dict]) -> dict:
        """Agrega scores individuales vía worst-message-wins (max).

        Args:
            messages: Lista de resultados de score().

        Returns:
            dict con aggregate_score siendo el máximo de los mensajes,
            y cada contribución del mensaje más riesgoso.
        """
        if not messages:
            return {
                "aggregate_score": 0.0,
                "sentiment_contribution": 0.0,
                "frustration_contribution": 0.0,
                "intent_contribution": 0.0,
            }

        worst = max(messages, key=lambda m: m["aggregate_score"])
        return dict(worst)

