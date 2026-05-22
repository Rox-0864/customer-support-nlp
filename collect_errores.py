"""
Recolector de errores de clasificación para ConversaAI.

Uso:
  pip install -r requirements-collect.txt
  python collect_errores.py

Escribí mensajes de soporte en ES/PT, el modelo clasifica,
y si está mal, corregís y se guarda para re-entrenar.
"""

import csv
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

ERRORS_FILE = "errores_reales.csv"
INTENTS = ["cancelacion", "consulta", "facturacion", "problema_tecnico", "reclamo", "resuelto"]
SENTIMENTS = ["positive", "negative"]

# --- Modelos ---
print("Cargando modelos...")

HF_INTENT = "Rosela/xlmr-intent-6clases"
tokenizer_intent = AutoTokenizer.from_pretrained(HF_INTENT)
model_intent = AutoModelForSequenceClassification.from_pretrained(HF_INTENT)
pipe_intent = pipeline("text-classification", model=model_intent, tokenizer=tokenizer_intent, device=-1)

HF_SENTIMENT = "data/models/xlmr-sentiment"
if os.path.exists(HF_SENTIMENT) and os.path.exists(f"{HF_SENTIMENT}/config.json"):
    tokenizer_sent = AutoTokenizer.from_pretrained(HF_SENTIMENT)
    model_sent = AutoModelForSequenceClassification.from_pretrained(HF_SENTIMENT)
    pipe_sent = pipeline("sentiment-analysis", model=model_sent, tokenizer=tokenizer_sent, device=-1)
else:
    pipe_sent = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=-1)

print("Modelos listos!")
print()

def detect_lang(text):
    es_keywords = ["el", "la", "los", "las", "que", "por", "para", "con", "una", "del", "como", "más", "pero", "este", "entre", "porque", "cuando", "muy", "sin", "sobre", "tengo", "hace", "todo", "bien", "hola", "gracias", "quisiera", "necesito", "podrían", "llevo", "pedido", "producto", "pago", "cuenta", "contraseña", "envío", "devolución", "reembolso"]
    pt_keywords = ["o", "a", "os", "as", "que", "por", "para", "com", "uma", "do", "como", "mais", "mas", "este", "entre", "porque", "quando", "muito", "sem", "sobre", "tenho", "faz", "tudo", "bem", "olá", "obrigado", "gostaria", "preciso", "poderiam", "levei", "pedido", "produto", "pagamento", "conta", "senha", "envio", "devolução", "reembolso"]
    text_lower = text.lower()
    es_count = sum(1 for w in es_keywords if w in text_lower)
    pt_count = sum(1 for w in pt_keywords if w in text_lower)
    return "pt" if pt_count > es_count else "es"

def classify_sentiment(text):
    result = pipe_sent(text)[0]
    label_raw = result["label"]
    conf = float(result["score"])
    if "LABEL" in label_raw:
        idx = int(label_raw.split("_")[1])
        label = "positive" if idx >= 3 else "negative"
    elif label_raw.lower() in ("positive", "negative"):
        label = label_raw.lower()
    else:
        try:
            star = int(label_raw[0])
            label = "positive" if star >= 4 else "negative"
        except:
            label = label_raw.lower()
    return label, conf

def classify_intent(text):
    result = pipe_intent(text)[0]
    return result["label"], float(result["score"])

# --- Main ---
os.makedirs(os.path.dirname(ERRORS_FILE) or ".", exist_ok=True)

print("=" * 50)
print("🧪 Recolector de errores - ConversaAI")
print("=" * 50)
print("Escribí mensajes de soporte en español o portugués.")
print("Si la clasificación es incorrecta, la corregís y se guarda.")
print()
print(f"Intents: {', '.join(INTENTS)}")
print(f"Sentimientos: {', '.join(SENTIMENTS)}")
print()

while True:
    text = input("\n📝 Mensaje (o 'q' para salir): ").strip()
    if text.lower() == "q":
        break
    if not text:
        continue

    lang = detect_lang(text)
    sent_label, sent_conf = classify_sentiment(text)
    intent_label, intent_conf = classify_intent(text)

    print(f"\n   🌐 Idioma detectado: {lang}")
    print(f"   😤 Sentimiento: {sent_label.upper()} ({sent_conf:.1%})")
    print(f"   🎯 Intención:   {intent_label} ({intent_conf:.1%})")

    correcto = input("\n   ¿Es correcto? (s/n): ").strip().lower()
    if correcto == "s":
        continue

    if correcto == "n":
        print(f"\n   Intents ({', '.join(INTENTS)})")
        intent_ok = input("   Intención correcta: ").strip()
        print(f"   Sentimientos ({', '.join(SENTIMENTS)})")
        sent_ok = input("   Sentimiento correcto: ").strip()

        if intent_ok in INTENTS and sent_ok in SENTIMENTS:
            es_nuevo = not os.path.exists(ERRORS_FILE)
            with open(ERRORS_FILE, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if es_nuevo:
                    w.writerow(["text", "intent", "sentiment", "lang"])
                w.writerow([text, intent_ok, sent_ok, lang])
            print(f"   ✅ Guardado en {ERRORS_FILE}")
        else:
            print("   ⚠️ Valores inválidos")

    print(f"\n{'=' * 50}")

total = 0
if os.path.exists(ERRORS_FILE):
    with open(ERRORS_FILE, encoding="utf-8") as f:
        total = sum(1 for _ in csv.DictReader(f))
print(f"\n📊 Total errores recolectados: {total}")
print(f"   Archivo: {ERRORS_FILE}")
