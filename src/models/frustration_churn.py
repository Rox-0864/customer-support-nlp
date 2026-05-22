"""
Frustration detection and churn prediction module.
Analyzes conversation threads to detect frustration patterns and predict churn risk.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

@dataclass
class MessageAnalysis:
    """Analysis of a single message in a conversation."""
    text: str
    sentiment: str  # positive, negative, neutral
    sentiment_confidence: float
    intent: str  # reclamo, consulta, cancelacion, problema_tecnico, facturacion, resuelto
    intent_confidence: float
    frustration_score: float  # 0.0 - 1.0
    timestamp: datetime

@dataclass
class ThreadAnalysis:
    """Analysis of a complete conversation thread."""
    thread_id: str
    messages: List[MessageAnalysis]
    overall_sentiment: str
    avg_frustration: float
    frustration_trend: str  # increasing, decreasing, stable
    churn_risk: str  # low, medium, high, critical
    unresolved_intents: List[str]
    frustration_peaks: List[int]  # message indices
    recommendation: str

# Frustration keywords and patterns
FRUSTRATION_PATTERNS = {
    "es": {
        "high": [
            r"ya (llamé|reporte|escribí) \d+ (veces|días|horas)",
            r"por (tercera|cuarta|quinta) vez",
            r"nunca me (ayudan|solucionan|responden)",
            r"es inaceptable",
            r"voy a cancelar",
            r"me voy a la competencia",
            r"pésimo|terrible|horrible",
            r"olvídense?|cancelen.*ya",
        ],
        "medium": [
            r"llevo \d+ (días|horas|semanas)",
            r"no funciona",
            r"no me (ayudan|explican)",
            r"(problema|error|falla)",
            r"tardaron mucho",
        ],
        "negative_intents": ["reclamo", "cancelacion", "problema_tecnico"],
        "resolution_intents": ["resuelto"]
    },
    "pt": {
        "high": [
            r"já (liguei|reportei|escrevi) \d+ (vezes|dias|horas)",
            r"pela (terceira|quarta|quinta) vez",
            r"nunca me (ajudam|resolvem|respondem)",
            r"é inaceitável",
            r"vou cancelar",
            r"vou para a concorrência",
            r"péssimo|terrível|horrível",
            r"esqueçam?|cancelem.*já",
        ],
        "medium": [
            r"já faz \d+ (dias|horas|semanas)",
            r"não funciona",
            r"não me (ajudam|explicam)",
            r"(problema|erro|falha)",
            r"demoraram muito",
        ],
        "negative_intents": ["reclamo", "cancelamento", "problema_tecnico"],
        "resolution_intents": ["resolvido"]
    }
}

def calculate_frustration_score(
    text: str, 
    sentiment: str, 
    intent: str, 
    lang: str,
    message_history: List[MessageAnalysis] = None
) -> float:
    """
    Calculate frustration score (0.0 - 1.0) for a message.
    
    Factors:
    - Sentiment (negative = higher)
    - Intent (reclamo/cancelacion = higher)
    - Frustration patterns in text
    - Escalation in conversation history
    """
    score = 0.0
    
    # Sentiment contribution (0-0.4)
    if sentiment == "negative":
        score += 0.4
    elif sentiment == "neutral":
        score += 0.2
    
    # Intent contribution (0-0.3)
    negative_intents = FRUSTRATION_PATTERNS[lang]["negative_intents"]
    if intent in negative_intents:
        if intent in ["cancelacion", "cancelamento"]:
            score += 0.3
        else:
            score += 0.2
    
    # Pattern matching (0-0.3)
    text_lower = text.lower()
    patterns = FRUSTRATION_PATTERNS[lang]
    
    for pattern in patterns["high"]:
        if re.search(pattern, text_lower):
            score += 0.3
            break
    
    if score < 0.3:  # Only check medium if not already high
        for pattern in patterns["medium"]:
            if re.search(pattern, text_lower):
                score += 0.15
                break
    
    # Escalation in history (0-0.2 bonus)
    if message_history and len(message_history) > 0:
        recent = message_history[-3:]  # Last 3 messages
        if len(recent) >= 2:
            avg_prev_frustration = sum(m.frustration_score for m in recent) / len(recent)
            if score > avg_prev_frustration + 0.2:
                score += 0.2  # Escalation bonus
    
    return min(score, 1.0)

def detect_frustration_trend(messages: List[MessageAnalysis]) -> str:
    """Detect if frustration is increasing, decreasing, or stable."""
    if len(messages) < 3:
        return "stable"
    
    # Split into halves and compare average frustration
    n = len(messages)
    first_half = messages[:n//2]
    second_half = messages[n//2:]
    
    avg_first = sum(m.frustration_score for m in first_half) / len(first_half)
    avg_second = sum(m.frustration_score for m in second_half) / len(second_half)
    
    diff = avg_second - avg_first
    
    if diff > 0.15:
        return "increasing"
    elif diff < -0.15:
        return "decreasing"
    else:
        return "stable"

def predict_churn_risk(
    messages: List[MessageAnalysis],
    thread_duration_hours: float
) -> str:
    """
    Predict churn risk based on conversation analysis.
    
    Returns: "low", "medium", "high", "critical"
    """
    if len(messages) == 0:
        return "low"
    
    # Calculate metrics
    avg_frustration = sum(m.frustration_score for m in messages) / len(messages)
    negative_count = sum(1 for m in messages if m.sentiment == "negative")
    negative_ratio = negative_count / len(messages)
    
    # Check for cancellation intent
    cancel_intents = ["cancelacion", "cancelamento"]
    has_cancel_intent = any(m.intent in cancel_intents for m in messages)
    
    # Check for resolution
    resolution_intents = FRUSTRATION_PATTERNS["es"]["resolution_intents"] + \
                        FRUSTRATION_PATTERNS["pt"]["resolution_intents"]
    has_resolution = any(m.intent in resolution_intents for m in messages)
    
    # Churn prediction logic
    if has_resolution and not has_cancel_intent:
        return "low"
    
    if has_cancel_intent:
        if avg_frustration > 0.7 or negative_ratio > 0.8:
            return "critical"
        else:
            return "high"
    
    if avg_frustration > 0.6 and negative_ratio > 0.7:
        return "high"
    elif avg_frustration > 0.4 or negative_ratio > 0.5:
        return "medium"
    else:
        return "low"

def find_frustration_peaks(messages: List[MessageAnalysis]) -> List[int]:
    """Find message indices where frustration peaks."""
    peaks = []
    window = 3
    
    for i in range(len(messages)):
        current = messages[i].frustration_score
        
        # Check if it's a local maximum in surrounding window
        is_peak = True
        for j in range(max(0, i - window), min(len(messages), i + window + 1)):
            if j != i and messages[j].frustration_score > current:
                is_peak = False
                break
        
        if is_peak and current > 0.5:  # Only consider significant peaks
            peaks.append(i)
    
    return peaks

def find_unresolved_intents(messages: List[MessageAnalysis]) -> List[str]:
    """Find intents that were never resolved."""
    intent_mentions = {}  # intent -> list of (index, resolved)
    
    resolution_intents = FRUSTRATION_PATTERNS["es"]["resolution_intents"] + \
                        FRUSTRATION_PATTERNS["pt"]["resolution_intents"]
    
    for i, msg in enumerate(messages):
        if msg.intent not in resolution_intents:
            if msg.intent not in intent_mentions:
                intent_mentions[msg.intent] = []
            intent_mentions[msg.intent].append((i, False))
        else:
            # Resolution message - mark previous intents as resolved
            for intent in intent_mentions:
                for j, (idx, resolved) in enumerate(intent_mentions[intent]):
                    if not resolved and idx < i:
                        intent_mentions[intent][j] = (idx, True)
    
    # Return intents that have no resolution
    unresolved = []
    for intent, mentions in intent_mentions.items():
        if any(not resolved for _, resolved in mentions):
            unresolved.append(intent)
    
    return unresolved

def generate_recommendation(analysis: ThreadAnalysis) -> str:
    """Generate actionable recommendation based on analysis."""
    recommendations = []
    
    if analysis.churn_risk == "critical":
        recommendations.append("🚨 ACCIÓN INMEDIATA: Contactar al cliente por canal prioritario")
        recommendations.append("Ofrecer compensación o escalamiento a nivel superior")
    elif analysis.churn_risk == "high":
        recommendations.append("⚠️ Alta probabilidad de abandono: Asignar agente senior")
        recommendations.append("Revisar y acelerar resolución de tickets abiertos")
    
    if "cancelacion" in analysis.unresolved_intents or "cancelamento" in analysis.unresolved_intents:
        recommendations.append("💼 Ofrecer retención proactiva con descuento o mejora")
    
    if analysis.frustration_trend == "increasing":
        recommendations.append("📈 Frustración en aumento: Revisar calidad de respuestas")
    
    if analysis.avg_frustration > 0.6:
        recommendations.append("🔴 Cliente muy frustrado: Considerar compensación")
    
    if not recommendations:
        if analysis.churn_risk == "low":
            recommendations.append("✅ Riesgo bajo: Monitorear conversación")
        else:
            recommendations.append("➡️ Seguir protocolo estándar de atención")
    
    return "\n".join(recommendations)

def analyze_thread(
    thread_id: str,
    messages: List[Dict],  # List of {text, sentiment, intent, timestamp, ...}
    lang: str = "es"
) -> ThreadAnalysis:
    """
    Analyze a complete conversation thread.
    
    Args:
        thread_id: Unique identifier for the thread
        messages: List of message dicts with text, sentiment, intent, timestamp
        lang: Language code ("es" or "pt")
    
    Returns:
        ThreadAnalysis object with complete analysis
    """
    # Convert to MessageAnalysis objects
    analyzed_messages = []
    msg_history = []
    
    for i, msg in enumerate(messages):
        # Calculate frustration score
        frustration = calculate_frustration_score(
            msg["text"],
            msg["sentiment"],
            msg["intent"],
            lang,
            msg_history if i > 0 else None
        )
        
        msg_analysis = MessageAnalysis(
            text=msg["text"],
            sentiment=msg["sentiment"],
            sentiment_confidence=msg.get("sentiment_confidence", 1.0),
            intent=msg["intent"],
            intent_confidence=msg.get("intent_confidence", 1.0),
            frustration_score=frustration,
            timestamp=msg.get("timestamp", datetime.now())
        )
        
        analyzed_messages.append(msg_analysis)
        msg_history.append(msg_analysis)
    
    # Overall sentiment
    pos_count = sum(1 for m in analyzed_messages if m.sentiment == "positive")
    neg_count = sum(1 for m in analyzed_messages if m.sentiment == "negative")
    overall = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
    
    # Average frustration
    avg_frustration = sum(m.frustration_score for m in analyzed_messages) / len(analyzed_messages)
    
    # Thread duration
    if len(analyzed_messages) > 1 and hasattr(analyzed_messages[0].timestamp, 'hour'):
        duration = (analyzed_messages[-1].timestamp - analyzed_messages[0].timestamp).total_seconds() / 3600
    else:
        duration = 1.0  # Default 1 hour
    
    # Analysis
    trend = detect_frustration_trend(analyzed_messages)
    churn = predict_churn_risk(analyzed_messages, duration)
    unresolved = find_unresolved_intents(analyzed_messages)
    peaks = find_frustration_peaks(analyzed_messages)
    
    # Create thread analysis
    analysis = ThreadAnalysis(
        thread_id=thread_id,
        messages=analyzed_messages,
        overall_sentiment=overall,
        avg_frustration=avg_frustration,
        frustration_trend=trend,
        churn_risk=churn,
        unresolved_intents=unresolved,
        frustration_peaks=peaks,
        recommendation=""
    )
    
    # Generate recommendation
    analysis.recommendation = generate_recommendation(analysis)
    
    return analysis
