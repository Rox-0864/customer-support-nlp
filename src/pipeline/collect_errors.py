import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import csv
from src.models.intent_classifier import classify_intent
from src.models.xlmr_sentiment import classify as classify_sentiment

ERRORS_FILE = "data/training/errores_reales.csv"
INTENTS = ["cancelacion", "consulta", "facturacion", "problema_tecnico", "reclamo", "resuelto"]
SENTIMENTS = ["positive", "negative"]

os.makedirs("data/training", exist_ok=True)

def save_error(text, intent_correcto, sentimiento_correcto, lang, intent_original, sent_original):
    es_nuevo = not os.path.exists(ERRORS_FILE)
    with open(ERRORS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if es_nuevo:
            w.writerow(["text", "intent", "sentiment", "lang"])
        w.writerow([text, intent_correcto, sentimiento_correcto, lang])
    print(f"   ✅ Guardado en {ERRORS_FILE}")

def main():
    print("=" * 50)
    print("🧪 Recolector de errores para re-entrenamiento")
    print("=" * 50)
    print("Escribí mensajes para probar. Si la clasificación es incorrecta,")
    print("te dejo corregirla y se guarda para re-entrenar.")
    print()
    print(f"Intents disponibles: {', '.join(INTENTS)}")
    print(f"Sentimientos: {', '.join(SENTIMENTS)}")
    print()

    while True:
        text = input("\n📝 Mensaje (o 'q' para salir): ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue

        sr = classify_sentiment(text)
        ir = classify_intent(text)

        print(f"\n   Sentimiento: {sr.label.upper()} ({sr.confidence:.1%})")
        print(f"   Intención:   {ir.intent} ({ir.confidence:.1%})")
        print(f"   Modelo:      {ir.model}")

        correcto = input("\n   ¿Es correcto? (s/n): ").strip().lower()
        if correcto == "s":
            continue
        if correcto == "n":
            print(f"\n   Intents: {', '.join(INTENTS)}")
            intent_ok = input("   Intención correcta: ").strip()
            print(f"   Sentimientos: {', '.join(SENTIMENTS)}")
            sent_ok = input("   Sentimiento correcto: ").strip()

            if intent_ok in INTENTS and sent_ok in SENTIMENTS:
                save_error(text, intent_ok, sent_ok, sr.lang, ir.intent, sr.label)
            else:
                print("   ⚠️ Valores inválidos, no se guardó")

        print(f"\n{'=' * 50}")

    total = 0
    if os.path.exists(ERRORS_FILE):
        with open(ERRORS_FILE) as f:
            total = sum(1 for _ in csv.DictReader(f))
    print(f"\n📊 Total errores recolectados: {total}")
    print(f"   Archivo: {ERRORS_FILE}")
    print("   Para re-entrenar: mergeá errores_reales.csv con bitext_intent_*.csv")

if __name__ == "__main__":
    main()
