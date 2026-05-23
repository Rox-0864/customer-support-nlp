"""
Recolector de errores de clasificación para ConversaAI.

Uso interactivo (default):
  python collect_errores.py

Uso batch (procesa CSVs enteros automáticamente):
  python collect_errores.py --batch bitext_intent_es.csv bitext_intent_pt.csv

En batch compara las predicciones contra las etiquetas originales
(mapeando 26 intents → 6 clases) y guarda solo los errores.

Trabajo en equipo:
  Persona 1 (ES): python collect_errores.py --batch bitext_intent_es.csv --output errores_es.csv
  Persona 2 (PT): python collect_errores.py --batch bitext_intent_pt.csv --output errores_pt.csv
  Para juntar:     python collect_errores.py --merge errores_es.csv errores_pt.csv --output errores_reales.csv
"""

import csv
import os
import sys
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

ERRORS_FILE = "errores_reales.csv"
INTENTS = ["cancelacion", "consulta", "facturacion", "problema_tecnico", "reclamo", "resuelto"]
SENTIMENTS = ["positive", "negative"]

# Mapping 26 intents originales → 6 clases
INTENT_MAPPING = {
    "cancel_order": "cancelacion",
    "check_cancellation_fee": "cancelacion",
    "complaint": "reclamo",
    "get_refund": "reclamo",
    "track_refund": "reclamo",
    "check_invoice": "facturacion",
    "get_invoice": "facturacion",
    "check_payment_methods": "facturacion",
    "payment_issue": "facturacion",
    "check_refund_policy": "facturacion",
    "recover_password": "problema_tecnico",
    "registration_problems": "problema_tecnico",
    "review": "resuelto",
    "contact_customer_service": "consulta",
    "contact_human_agent": "consulta",
    "delivery_options": "consulta",
    "delivery_period": "consulta",
    "change_order": "consulta",
    "place_order": "consulta",
    "track_order": "consulta",
    "change_shipping_address": "consulta",
    "set_up_shipping_address": "consulta",
    "create_account": "consulta",
    "delete_account": "consulta",
    "edit_account": "consulta",
    "switch_account": "consulta",
    "newsletter_subscription": "consulta",
}

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
        except Exception:
            label = label_raw.lower()
    return label, conf


def classify_intent(text):
    result = pipe_intent(text)[0]
    return result["label"], float(result["score"])


def save_error(text, intent, sentiment, lang):
    es_nuevo = not os.path.exists(ERRORS_FILE)
    with open(ERRORS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if es_nuevo:
            w.writerow(["text", "intent", "sentiment", "lang"])
        w.writerow([text, intent, sentiment, lang])


def run_batch(csv_paths, output_file):
    global ERRORS_FILE
    ERRORS_FILE = output_file
    total = 0
    errores = 0

    for path in csv_paths:
        if not os.path.exists(path):
            print(f"   ⚠️ No encontrado: {path}")
            continue

        lang = "pt" if "pt" in path else "es"
        print(f"\n📂 Procesando: {path} ({lang.upper()})")

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col_intent = "intent" if "intent" in reader.fieldnames else "intencion"
            col_text = "text" if "text" in reader.fieldnames else "mensaje_cliente"

            rows = list(reader)
            print(f"   Total: {len(rows)} filas")

            for row in rows:
                text = row[col_text].strip()
                expected_26 = row[col_intent].strip()
                expected_6 = INTENT_MAPPING.get(expected_26, expected_26)

                if expected_6 not in INTENTS:
                    continue

                pred_intent, pred_conf = classify_intent(text)
                pred_sent, _ = classify_sentiment(text)

                if pred_intent != expected_6:
                    save_error(text, expected_6, pred_sent, lang)
                    errores += 1

                total += 1

    print(f"\n{'=' * 50}")
    print(f"📊 Resultados batch")
    print(f"{'=' * 50}")
    print(f"   Procesados: {total}")
    print(f"   Errores guardados: {errores}")
    print(f"   Precisión actual: {(1 - errores / total) * 100:.1f}%" if total > 0 else "   Sin datos")
    print(f"   Archivo: {ERRORS_FILE}")
    print(f"\n📥 Compartí {ERRORS_FILE} para re-entrenar el modelo.")


def merge(input_files, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["text", "intent", "sentiment", "lang"])
        for fname in input_files:
            with open(fname, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    w.writerow([row["text"], row["intent"], row["sentiment"], row["lang"]])
    print(f"✅ Merge completado: {output_file} ({sum(1 for _ in open(output_file)) - 1} errores)")


def run_interactive():
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
                save_error(text, intent_ok, sent_ok, lang)
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


# --- Entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolector de errores ConversaAI")
    parser.add_argument("--batch", nargs="+", help="CSVs a procesar en batch (ES y PT)")
    parser.add_argument("--merge", nargs="+", help="Archivos de errores a mergear")
    parser.add_argument("--output", default=ERRORS_FILE, help="Archivo de salida (default: errores_reales.csv)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.merge:
        merge(args.merge, args.output)
    elif args.batch:
        run_batch(args.batch, args.output)
    else:
        run_interactive()
