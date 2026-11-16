import os
import re
import sys
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.churn import ChurnScorer

SENTIMENT_KEYWORDS = {
    "positive": {
        "es": [
            "excelente", "genial", "maravilloso", "fantástico", "bueno", "buenísimo",
            "estupendo", "perfecto", "increíble", "me encantó", "me gustó", "feliz",
            "contento", "satisfecho", "rápido", "eficiente", "recomiendo", "gracias",
            "agradezco", "espectacular", "magnífico",             "nota 10", "felicidades",
            "encantado", "alegre", "bien", "mejor", "excelente atención", "resolvieron",
        ],
        "pt": [
            "excelente", "genial", "maravilhoso", "fantástico", "bom", "ótimo",
            "perfeito", "incrível", "adorei", "gostei", "feliz", "contente",
            "satisfeito", "rápido", "eficiente", "recomendo", "obrigado", "obrigada",
            "agradeço", "espetacular", "magnífico",             "nota 10", "parabéns",
            "encantado", "alegre", "bem", "melhor", "resolveram",
        ],
    },
    "negative": {
        "es": [
            "pésimo", "terrible", "horrible", "malo", "malísimo", "pésimo servicio",
            "nunca más", "indignante", "inaceptable", "vergüenza", "decepcionado",
            "frustrado", "enojado", "molesto", "lento", "tardó", "demoró",
            "no funciona", "no sirve", "roto", "error", "falla", "problema",
            "queja", "reclamo", "pésima atención", "pessimo", "horroroso",
            "odio", "no me responde",
            "no responden", "estafa", "timo", "robo",
        ],
        "pt": [
            "péssimo", "terrível", "horrível", "ruim", "péssimo atendimento",
            "nunca mais", "indignante", "inaceitável", "vergonha", "decepcionado",
            "frustrado", "irritado", "chateado", "lento", "demorou", "atrasou",
            "não funciona", "não serve", "quebrado", "erro", "falha", "problema",
            "reclamação", "péssimo", "horroroso", "odeio", "não responde",
            "não respondem", "golpe", "fraude",
        ],
    },
    "neutral": {
        "es": [
            "consulta", "pregunta", "información", "duda", "quiero saber",
            "necesito saber", "me gustaría saber", "podrían decirme",
            "cuál es", "cómo es", "qué es", "dónde está", "cuándo",
            "horario", "dirección", "teléfono", "precio", "valor",
            "costo", "disponibilidad", "plazo", "entrega",
        ],
        "pt": [
            "consulta", "pergunta", "informação", "dúvida", "quero saber",
            "preciso saber", "gostaria de saber", "poderiam me dizer",
            "qual é", "como é", "o que é", "onde fica", "quando",
            "horário", "endereço", "telefone", "preço", "valor",
            "custo", "disponibilidade", "prazo", "entrega",
        ],
    },
}

INTENT_PATTERNS: Dict[str, List[Dict[str, List[str]]]] = {
    "cancelacion": [
        {"es": [r"cancel", r"cancelaci[oó]n", r"anul", r"baja", r"cancelamento", r"desuscrib"]},
        {"pt": [r"cancel", r"cancelamento", r"anul", r"baixa", r"cancelar", r"desinscrev"]},
    ],
    "consulta_general": [
        {"es": [r"consulta", r"pregunt", r"duda", r"informaci[oó]n", r"qu[ie]ro saber",
                r"necesito saber", r"c[oó]mo funcion", r"qu[eé] es", r"qu[eé] significa"]},
        {"pt": [r"consulta", r"pergunta", r"d[uú]vida", r"informa[cç][aã]o", r"quero saber",
                r"preciso saber", r"como funcion", r"o que é", r"o que significa"]},
    ],
    "facturacion_pago": [
        {"es": [r"factur", r"pago", r"cobr", r"facturaci[oó]n", r"boleto", r"recibo",
                r"comprobante", r"pag[uú]e", r"deuda", r"saldo", r"vencimient"]},
        {"pt": [r"fatura", r"pagamento", r"cobran[cç]a", r"fatura[cç][aã]o", r"boleto",
                r"recibo", r"comprovante", r"paguei", r"d[vv]ida", r"saldo", r"vencimento"]},
    ],
    "feedback": [
        {"es": [r"feedback", r"opini[oó]n", r"sugerencia", r"recomendaci[oó]n", r"me gust",
                r"me encant", r"excelente", r"genial", r"mejor[ée]",
                r"quiero opinar", r"valoraci[oó]n"]},
        {"pt": [r"feedback", r"opini[aã]o", r"sugest[aã]o", r"recomenda[cç][aã]o", r"gostei",
                r"adorei", r"excelente", r"genial", r"melhor", r"quero opinar", r"avalia[cç][aã]o"]},
    ],
    "gestion_cuenta": [
        {"es": [r"cambiar mi", r"actualizar mi", r"modificar mi", r"cuenta", r"datos",
                r"contraseñ", r"correo", r"email", r"direcci[oó]n", r"tel[eé]fono",
                r"usuario", r"registr", r"perfil"]},
        {"pt": [r"cambiar m", r"atualizar m", r"modificar m", r"conta", r"dados",
                r"senha", r"correio", r"email", r"endere[çc]o", r"telefone",
                r"usu[aá]rio", r"registr", r"perfil"]},
    ],
    "modificacion_pedido": [
        {"es": [r"modificar mi pedido", r"cambiar mi pedido", r"actualizar mi pedido",
                r"pedido", r"compro", r"compra", r"orden", r"cambiar talla",
                r"cambiar color", r"cambiar direcci[oó]n", r"cancelar pedido",
                r"modificar pedido"]},
        {"pt": [r"modificar meu pedido", r"alterar meu pedido", r"atualizar meu pedido",
                r"pedido", r"comprei", r"compra", r"ordem", r"trocar tamanho",
                r"trocar cor", r"mudar endere[çc]o", r"cancelar pedido",
                r"modificar pedido"]},
    ],
    "queja": [
        {"es": [r"queja", r"reclamo", r"indignad", r"inaceptable", r"verguenza",
                r"p[eé]simo", r"terrible", r"horrible", r"estafa", r"timo",
                r"quiero hablar con", r"voy a denunciar", r"abuso"]},
        {"pt": [r"queixa", r"reclama[cç][aã]o", r"indignad", r"inaceit[aá]vel", r"vergonha",
                r"p[eé]ssimo", r"terr[ií]vel", r"horr[ií]vel", r"golpe", r"fraude",
                r"quero falar com", r"vou denunciar", r"abuso"]},
    ],
    "reembolso": [
        {"es": [r"reembols", r"devolver", r"devoluci[oó]n", r"mi dinero", r"mi plata",
                r"reembolso", r"reembolsar", r"recoger", r"garant[ií]a",
                r"quiero que me devuelvan", r"que me devuelvan"]},
        {"pt": [r"reembols", r"devolver", r"devolu[cç][aã]o", r"meu dinheiro", r"reembolso",
                r"reembolsar", r"recolher", r"garantia", r"quero que me devolvam",
                r"que me devolvam", r"estorno"]},
    ],
    "seguimiento": [
        {"es": [r"seguimient", r"estado de mi", r"d[oó]nde est", r"c[oó]mo va mi",
                r"c[oó]mo viene mi", r"tracking", r"rastre", r"seguimiento",
                r"n[uú]mero de pedido", r"c[oó]digo de", r"cu[aá]ndo lleg",
                r"ya deber[ií]a"]},
        {"pt": [r"acompanhament", r"status do meu", r"onde est[aá]", r"como vai meu",
                r"como vem meu", r"tracking", r"rastrei", r"acompanhamento",
                r"n[úu]mero do pedido", r"c[oó]digo de", r"quando cheg",
                r"j[aá] deveria"]},
    ],
}

def detect_language(text: str) -> str:
    text_lower = text.lower()
    pt_markers = ["ão", "çã", "õe", "você", "vc", "obrigad", "gostari", "adorei",
                  "pessimo", "horrivel", "não", "obrigado", "obrigada"]
    score = sum(1 for m in pt_markers if m in text_lower)
    return "pt" if score >= 2 else "es"

class FallbackPipeline:
    def __init__(self):
        self._status = {"sentiment": "fallback", "intent": "fallback"}
        self._churn_scorer = ChurnScorer()

    def predict_sentiment(self, text: str) -> dict:
        if not text or not text.strip():
            return {"label": "neutral", "probability": 1.0,
                    "probabilities": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}}

        lang = detect_language(text)
        text_lower = text.lower()

        pos_keywords = SENTIMENT_KEYWORDS["positive"].get(lang, SENTIMENT_KEYWORDS["positive"]["es"])
        neg_keywords = SENTIMENT_KEYWORDS["negative"].get(lang, SENTIMENT_KEYWORDS["negative"]["es"])
        neu_keywords = SENTIMENT_KEYWORDS["neutral"].get(lang, SENTIMENT_KEYWORDS["neutral"]["es"])

        NEGATION_WORDS = {
            "es": ["no ", "nunca ", "tampoco ", "jamás ", "ni "],
            "pt": ["não ", "nunca ", "tampouco ", "jamais ", "nem "],
        }
        negators = NEGATION_WORDS.get(lang, NEGATION_WORDS["es"])

        def count_kw(kws, text, negators):
            c = 0
            for kw in kws:
                idx = text.find(kw)
                if idx == -1:
                    continue
                preceding = text[max(0, idx - 40):idx]
                if not any(n in preceding for n in negators):
                    c += 1
            return c

        pos_count = count_kw(pos_keywords, text_lower, negators)
        neg_count = count_kw(neg_keywords, text_lower, negators)
        neu_count = count_kw(neu_keywords, text_lower, negators)

        total = pos_count + neg_count + neu_count
        if total == 0:
            return {"label": "neutral", "probability": 0.6,
                    "probabilities": {"negative": 0.2, "neutral": 0.6, "positive": 0.2}}

        pos_prob = pos_count / total
        neg_prob = neg_count / total
        neu_prob = neu_count / total

        if pos_prob >= neg_prob and pos_prob >= neu_prob:
            label = "positive"
            prob = pos_prob
        elif neg_prob >= pos_prob and neg_prob >= neu_prob:
            label = "negative"
            prob = neg_prob
        else:
            label = "neutral"
            prob = neu_prob

        return {
            "label": label,
            "probability": prob,
            "probabilities": {"negative": neg_prob, "neutral": neu_prob, "positive": pos_prob},
        }

    def predict_intent(self, text: str) -> dict:
        if not text or not text.strip():
            probs = {intent: 0.0 for intent in INTENT_PATTERNS}
            probs["consulta_general"] = 1.0
            return {"intent": "consulta_general", "probability": 1.0, "probabilities": probs}

        lang = detect_language(text)
        text_lower = text.lower()

        scores = {}
        for intent, patterns_list in INTENT_PATTERNS.items():
            score = 0.0
            for pattern_group in patterns_list:
                patterns = pattern_group.get(lang, [])
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        score += 1.0
            scores[intent] = score

        total = sum(scores.values())
        if total <= 0:
            best_intent = "consulta_general"
            best_prob = 0.3
            probabilities = {intent: 0.0 for intent in INTENT_PATTERNS}
            probabilities["consulta_general"] = 1.0
        else:
            best_intent = max(scores, key=scores.get)
            best_prob = min(scores[best_intent] / total, 0.95)
            probabilities = {intent: scores[intent] / total for intent in INTENT_PATTERNS}

        return {"intent": best_intent, "probability": best_prob, "probabilities": probabilities}

    def compute_churn(self, text: str, sentiment: dict, intent: dict) -> dict:
        """Calcula el churn con la misma fórmula que el pipeline principal (ChurnScorer)."""
        return self._churn_scorer.score(sentiment, intent, text)

    def predict(self, text: str) -> dict:
        sentiment = self.predict_sentiment(text)
        intent = self.predict_intent(text)
        churn = self.compute_churn(text, sentiment, intent)
        return {"text": text, "sentiment": sentiment, "intent": intent, "churn": churn}

    def batch_predict(self, texts: List[str]) -> List[dict]:
        return [self.predict(t) for t in texts]

    def get_model_status(self) -> dict:
        return dict(self._status)
