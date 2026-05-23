import pandas as pd
import os
import sys
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "data/models/xlmr-sentiment"
CSV_PATH = "data/support_conversations.csv"
OUTPUT_PATH = "data/conversations_classified.csv"
BATCH_SIZE = 256

print(f"Cargando modelo desde {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1, batch_size=BATCH_SIZE)
print("Modelo listo.\n")

df = pd.read_csv(CSV_PATH)
print(f"Clasificando {len(df)} mensajes...")

texts = df['text'].astype(str).tolist()

results = pipe(texts)
sentiments = []
confs = []

for r in results:
    label_raw = r["label"]
    if "LABEL" in label_raw:
        idx = int(label_raw.split("_")[1])
        label = "positive" if idx == 1 else "negative"
    else:
        label = label_raw.lower()
    sentiments.append(label)
    confs.append(float(r["score"]))

df['sentiment'] = sentiments
df['sentiment_confidence'] = confs

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nGuardado en {OUTPUT_PATH}")
print(df.groupby('sentiment').size().to_string())
