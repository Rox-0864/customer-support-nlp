# ConversaAI — Sistema de Análisis de Conversaciones de Soporte

[![CI — Tests](https://github.com/Rox-0864/customer-support-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/Rox-0864/customer-support-nlp/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Hugging Face](https://img.shields.io/badge/🤗-Models%20on%20HF%20Hub-yellow)](https://huggingface.co/Rosela)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://rosela-conversaai-dashboard.hf.space)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-84%20passing-brightgreen)](tests/)

Pipeline de NLP para clasificar **sentimiento** (3 clases), **intención** (9 clases) y **riesgo de churn** en conversaciones de soporte al cliente en español y portugués.

## Enlaces

| Recurso | URL |
|---------|-----|
| Repositorio GitHub | [Rox-0864/customer-support-nlp](https://github.com/Rox-0864/customer-support-nlp) |
| Modelo Sentimiento (HF Hub) | [Rosela/xlm-r-sentiment-espt](https://huggingface.co/Rosela/xlm-r-sentiment-espt) |
| Modelo Intención (HF Hub) | [Rosela/xlm-r-intent-espt](https://huggingface.co/Rosela/xlm-r-intent-espt) |
| Perfil HF Hub | [huggingface.co/Rosela](https://huggingface.co/Rosela) |
| Dashboard Streamlit (HF Space) | [rosela-conversaai-dashboard.hf.space](https://rosela-conversaai-dashboard.hf.space) |

## Arquitectura

```
Mensaje
  ├──► SentimentClassifier (XLM-R fine-tuneado)
  │     3 clases: negative / neutral / positive
  ├──► IntentClassifier (XLM-R fine-tuneado)
  │     9 clases: cancelacion, consulta_general, facturacion_pago,
  │               feedback, gestion_cuenta, modificacion_pedido,
  │               queja, reembolso, seguimiento
  └──► ChurnScorer (heurístico)
        w1 * sent_prob_negative + w2 * frust_flag + w3 * max(intent_risk)
        Pesos default: w1=0.3, w2=0.35, w3=0.35
```

### Fallback automático

Si los modelos fine-tuneados no están disponibles (sin GPU/local), el pipeline cae a:
- **Sentimiento**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Intención**: `MoritzLaurer/DeBERTa-v3-base-mnli` (zero-shot)
- **Demo offline**: `scripts/pipeline_fallback.py` — rule-based por keywords + regex (sin dependencias ML)

## Stack

| Componente | Tecnología |
|------------|-----------|
| Modelos | XLM-R (`xlm-roberta-base`) fine-tuneado ES/PT |
| Entrenamiento | Kaggle (GPU T4x2) + Hugging Face Hub |
| Dashboard | Streamlit + Plotly |
| Infraestructura | HuggingFace Spaces + Docker (Python 3.11-slim, CPU) |
| Tests | pytest (61 tests) + GitHub Actions CI |

## Estructura del proyecto

```
├── app.py                         # Dashboard Streamlit (HF Space)
├── src/
│   ├── pipeline.py                 # Orquestador ConversaAIPipeline
│   ├── data/processor.py           # DataProcessor: normalización + tokenización
│   ├── models/
│   │   ├── sentiment.py            # SentimentClassifier (XLM-R + fallback)
│   │   └── intent.py               # IntentClassifier (XLM-R + fallback)
│   └── features/
│       ├── churn.py                # ChurnScorer heurístico
│       └── business_rules.py       # Taxonomía + reglas de negocio unificadas
├── scripts/
│   └── pipeline_fallback.py        # Pipeline rule-based offline
├── notebooks/
│   ├── sentiment_xlmr.ipynb.py     # Fine-tuning sentimiento
│   └── intent_xlmr.ipynb.py        # Fine-tuning intención
├── data/
│   └── corpus_demo_bilingue.csv    # 108 mensajes demo ES/PT
├── tests/
│   ├── test_churn.py               # 19 tests — ChurnScorer
│   ├── test_pipeline_fallback.py   # 33 tests — FallbackPipeline
│   ├── test_processor.py           # 7 tests — DataProcessor
│   ├── test_business_rules.py      # 22 tests — reglas de negocio
│   └── test_pipeline_batch.py      # 3 tests — batch predict
├── requirements.txt
├── Dockerfile
└── .github/workflows/
    ├── ci.yml                      # CI: tests en push/PR
    └── pages.yml                   # Deploy dashboard-web a Pages
```

## Tests

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest tests/ -v
```

84 tests que cubren:
- **ChurnScorer**: scoring, pesos, detección de frustración ES/PT, conversaciones
- **FallbackPipeline**: sentimiento por keywords, intención por regex, detección de idioma, churn, batch predict
- **DataProcessor**: normalización de texto, unicode, espacios
- **BusinessRules**: overrides de intención/sentimiento, taxonomía, categorización de churn
- **Pipeline batch**: inferencia en batch real, preservación de orden, manejo de vacíos

## Dashboards

### Streamlit (HuggingFace Space)
👉 [rosela-conversaai-dashboard.hf.space](https://rosela-conversaai-dashboard.hf.space)

`app.py` — Dashboard único consolidado con navegación por páginas (Dashboard, Explorar, Acciones):
- Métricas principales y distribución de sentimiento/intención
- Explorador de mensajes con filtros y descarga CSV
- Histograma de churn interactivo y tabla de alto riesgo
- Stacked bar intención vs sentimiento y evolución temporal
- Recomendaciones para el equipo de producto y alertas

### Dashboard Web (Vercel)
👉 [s04-26-equipo-40-data-science.vercel.app](https://s04-26-equipo-40-data-science.vercel.app)

Dashboard web estático (HTML/CSS/JS) deployado en Vercel desde `dashboard-web/`.

## Docker

```bash
docker build -t conversaai .
docker run -p 8501:8501 conversaai
```

Los modelos fine-tuneados se descargan de HF Hub durante el build (~556MB c/u).

## Dataset demo

`data/corpus_demo_bilingue.csv` — 108 mensajes bilingües ES/PT etiquetados para prueba.
Para cargarlos en el dashboard, seleccioná "Demo (108 mensajes)" en el sidebar.

## Modelos fine-tuneados (HF Hub)

| Modelo | Clases | Accuracy | F1 Macro |
|--------|--------|----------|----------|
| [xlm-r-sentiment-espt](https://huggingface.co/Rosela/xlm-r-sentiment-espt) | negative, neutral, positive | 92.2% | 0.92 |
| [xlm-r-intent-espt](https://huggingface.co/Rosela/xlm-r-intent-espt) | 9 intenciones | 97.85% | 0.97 |

## Dataset fine-tuning (Kaggle)

Dataset en Kaggle: [rosaiselagonzalez/conversaai-sentiment-data](https://www.kaggle.com/datasets/rosaiselagonzalez/conversaai-sentiment-data)

## Licencia

MIT
