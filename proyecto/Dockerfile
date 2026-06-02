FROM python:3.11-slim

WORKDIR /app

# ── Dependencias ──
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# ── Pre-descargar modelos fine-tuneados de HF Hub (~556MB c/u) ──
# Se cachean en /root/.cache/huggingface/ durante el build.
# Así el container arranca al toque sin tener que descargar en runtime.
RUN python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print('⏬ Descargando modelo de sentimiento...')
AutoTokenizer.from_pretrained('Rosela/xlm-r-sentiment-espt')
AutoModelForSequenceClassification.from_pretrained('Rosela/xlm-r-sentiment-espt')
print('⏬ Descargando modelo de intención...')
AutoTokenizer.from_pretrained('Rosela/xlm-r-intent-espt')
AutoModelForSequenceClassification.from_pretrained('Rosela/xlm-r-intent-espt')
print('✅ Modelos precargados en caché')
" && rm -rf /root/.cache/pip

# ── Aplicación ──
COPY . .

# ── Limpiar artefactos de build ──
RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python3 -c 'import urllib.request; urllib.request.urlopen("http://localhost:8501/_stcore/health")'

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
