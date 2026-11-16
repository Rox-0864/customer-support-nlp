# 📚 Apuntes — ConversaAI

> Guía de estudio del pipeline de NLP, dashboard, y decisiones de arquitectura.
> Creada el 30/05/2026 · Actualizada el 16/11/2026 (refactor de unificación: reglas de negocio, churn y dashboards).

---

## 1. Arquitectura General

**¿Qué hace ConversaAI?**

Pipeline de NLP que:
1. Toma texto de conversaciones de soporte al cliente (ES/PT)
2. Clasifica sentimiento (3 clases) con XLM-RoBERTa
3. Clasifica intención (9 clases) con XLM-RoBERTa
4. Calcula riesgo de churn con reglas heurísticas
5. Muestra todo en un dashboard Streamlit con gráficos Plotly

```
Mensaje → XLM-R (sentimiento) → 3 clases
        → XLM-R (intención)     → 9 clases → Heurística de churn → Dashboard
```

**Stack:**
- Modelos: XLM-R (`xlm-roberta-base`) fine-tuneado ES/PT
- Dashboard: Streamlit + Plotly
- Infraestructura: HuggingFace Spaces (SDK streamlit)

---

## 2. Reglas de Negocio Heurísticas (Churn)

**¿Qué es una heurística?**

Regla simple basada en experiencia, NO un modelo ML. Son IF/ELSE que codifican conocimiento de negocio. Es un MVP — si en el futuro tienen datos de churn real, se reemplaza por un clasificador entrenado.

### Cálculo de riesgo de churn (`ChurnScorer` + `categorize_churn`)

El churn se calcula en **una sola fórmula** (`src/features/churn.py`), compartida por el pipeline principal y el fallback:

```python
# ChurnScorer.score() — score continuo en [0, 1]
churn = w1 * sent_prob_negative + w2 * frust_flag + w3 * max(intent_risk_probs)
# Pesos default: w1=0.3, w2=0.35, w3=0.35
```

Luego el score se convierte a categoría de negocio con `categorize_churn()` (`src/features/business_rules.py`):

```python
def categorize_churn(score, intent=None):
    if intent == "delete_account":        # 🔴 Eliminar cuenta = riesgo máximo siempre
        return "Alto Riesgo"
    if score >= 0.5:                      # 🔴 Alto
        return "Alto Riesgo"
    if score >= 0.25:                     # 🟠 Medio
        return "Riesgo Medio"
    return "Riesgo Bajo"                  # 🟢 Bajo
```

Los umbrales (`0.5` / `0.25`) están definidos como constantes `CHURN_HIGH_THRESHOLD` / `CHURN_MEDIUM_THRESHOLD`.

**¿Por qué no un modelo de ML para churn?**
1. Faltan datos etiquetados de churn real (necesitarías conectar con datos de retención post-venta)
2. Las reglas son transparentes: cualquier stakeholder entiende por qué alguien está en Alto Riesgo

---

## 3. Regla de Modificación de Pedido

**Problema**: El modelo de ML no siempre detecta cuando un usuario quiere modificar un pedido (ej: "pedí la roja pero necesito la azul").

Esta regla vive en `src/features/business_rules.py` como `MODIFY_ORDER_RULES`:

```python
MODIFY_ORDER_RULES = [
    (("cambi", "modific"), ("talla", "color", "pedido")),   # (grupo AND, grupo OR)
    (("pedí",), ("necesito",)),
]
```

- Si se detecta modificación, baja el sentimiento a Neutral para no sesgar métricas
- Una modificación no es necesariamente una experiencia negativa

---

## 4. Regla de Eliminación de Cuenta

**Problema**: El modelo de 9 clases NO incluye `delete_account`. Sin esta regla, "quiero borrar mi cuenta" se clasifica como `gestion_cuenta` o `consulta_general`, perdiendo la señal de churn más crítica.

Esta regla vive en `src/features/business_rules.py` como `DELETE_ACCOUNT_KEYWORDS` + la condición de que aparezca "cuenta"/"conta":

```python
DELETE_ACCOUNT_KEYWORDS = [
    'excluir', 'apagar minha conta', 'cancelar conta',
    'borrar mi cuenta', 'cancelar mi cuenta', 'eliminar',
    'elimino mi cuenta', 'cerrar mi cuenta', 'dar de baja', 'borrar cuenta'
]
# Condición: any(kw in texto) AND ("cuenta" in texto OR "conta" in texto)
```

Incluye palabras en español y portugués. `delete_account` es una intención **sintética** (no es clase del modelo), definida como `DELETE_ACCOUNT_INTENT` en el mismo módulo.

---

## 5. Taxonomía de Intenciones (unificada)

**Contexto:** antes existía un `INTENT_LABEL_MAP` que traducía las etiquetas del modelo (español) a labels del dashboard (inglés): `cancelacion → cancel_order`, `queja → complaint`, etc. Eso generaba **dos taxonomías paralelas** y era fuente de bugs.

**Solución (refactor):** se unificó todo a la taxonomía **española** canónica del modelo. Hoy hay una única fuente de verdad en `src/features/business_rules.py`:

```python
INTENT_LABELS = [
    "cancelacion", "consulta_general", "facturacion_pago", "feedback",
    "gestion_cuenta", "modificacion_pedido", "queja", "reembolso", "seguimiento",
]
DELETE_ACCOUNT_INTENT = "delete_account"   # sintética, solo la generan las reglas
```

| Intención | Macro-intención | Nombre UI |
|-----------|----------------|-----------|
| cancelacion | Gestión de Pedidos | Cancelación |
| modificacion_pedido | Gestión de Pedidos | Modificación Pedido |
| seguimiento | Gestión de Pedidos | Seguimiento |
| reembolso | Pagos y Reembolsos | Reembolso |
| facturacion_pago | Pagos y Reembolsos | Facturación/Pago |
| gestion_cuenta | Gestión de Cuenta | Gestión de Cuenta |
| delete_account | Gestión de Cuenta | Eliminación de Cuenta |
| queja | Quejas y Reclamos | Queja |
| feedback | Feedback | Feedback |
| consulta_general | Atención al Cliente | Consulta General |

Los mapeos viven como `MACRO_INTENT_MAP`, `INTENT_DISPLAY` y las funciones `macro_intent()` / `display_intent()` en el mismo módulo.

---

## 6. Soporte de Portugués

**¿El modelo funciona con portugués?** Sí. XLM-R es multilingüe y el modelo se entrenó con ES + PT (por eso el nombre termina en `-espt`). Acepta texto en portugués y etiqueta correctamente, pero las etiquetas de salida SIEMPRE están en español:

| Texto PT | Label que predice |
|----------|-------------------|
| "Quero cancelar meu pedido" | cancelacion |
| "Recebi o produto errado" | queja |
| "Como faço para trocar a cor?" | modificacion_pedido |

El idioma de cada mensaje se detecta aparte (para la gráfica de evolución temporal), pero la taxonomía de intenciones es única e independiente del idioma.

**Nota (resuelto en refactor)**: antes las labels solo estaban en español y se traducían con un mapa ES→EN. Eso ya no existe — se unificó todo a la taxonomía española canónica del modelo.

---

## 7. Carga de Modelos con Caché

```python
@st.cache_resource
def get_pipeline():
    return ConversaAIPipeline()   # Orquestador de src/pipeline.py

pipeline = get_pipeline()
```

- `@st.cache_resource` es un decorador de Streamlit que hace que la función se ejecute **una sola vez** y cachee el resultado. No importa cuántas veces recargues la página o interactúes — el pipeline se carga UNA VEZ.
- Los modelos se descargan de HF Hub (~556MB c/u).
- Internamente, `ConversaAIPipeline` instancia `SentimentClassifier` e `IntentClassifier`, que usan `AutoModelForSequenceClassification.from_pretrained()` + `AutoTokenizer.from_pretrained()`:
  1. Descarga config.json, tokenizer, y pesos desde HF Hub
  2. Crea el tokenizer (XLM-R usa SentencePiece, un tokenizer basado en subword/byte-level BPE)
  3. Carga los pesos en la arquitectura correcta
  4. Hace `tokenize → forward pass → softmax` para producir probabilidades por clase

### ¿Qué contiene cada modelo?

Cuando hacés fine-tuning con `Trainer` de HF y subís a Hub, se genera esta estructura:

```
xlm-r-intent-espt/              (igual para xlm-r-sentiment-espt)
├── config.json                  ← Arquitectura: 12 capas, 768 hidden size,
│                                   12 cabezas de atención, dropout, etc.
├── model.safetensors            ← Los pesos del modelo (~556MB). Es el
│                                   "cerebro" — sin esto no hay modelo.
├── tokenizer.json               ← Tokenizer SentencePiece (byte-level BPE)
├── tokenizer_config.json        ← Config del tokenizer (max_length, truncation)
├── special_tokens_map.json      ← Tokens especiales: [CLS], [SEP], [PAD], etc.
├── training_args.bin            ← Hiperparámetros usados: batch_size, lr,
│                                   epochs, warmup, etc. (solo referencia)
├── vocab.json                   ← Vocabulario completo (no siempre presente)
├── added_tokens.json            ← Tokens adicionales agregados al fine-tune
└── README.md                    ← Card del modelo en HF Hub
```

**`model.safetensors`** es el archivo crítico: contiene los 278 millones de parámetros de XLM-R guardados en float32. Sin él, el modelo no existe.

### ¿Por qué accuracy similar con 6000 vs 24000 datos?

**Punto clave**: XLM-R ya viene pre-entrenado con ~2.5TB de texto multilingüe. Vos no estás entrenando desde cero, estás haciendo **fine-tuning** — agarrás un modelo que ya sabe de lenguaje y le enseñás tu tarea específica.

Con 6000 mensajes bien balanceados, el modelo ya ve suficientes ejemplos por clase (~650 c/u). El fine-tuning converge rápido porque las representaciones de XLM-R ya son muy buenas.

| Aspecto | 6000 datos | 24000 datos |
|---------|-----------|-------------|
| Accuracy | ~97% | ~98% |
| Robustez (casos edge) | Media | Alta |
| Overfitting | Más riesgo | Menos riesgo |
| Code-switching ES/PT | Captura poco | Captura mejor |
| Tiempo de entrenamiento | ~20 min | ~1.5 h |

**La diferencia real no está en accuracy promedio**: en un test set estándar ambos dan números similares. La diferencia está en los **bordes** — con 24000 el modelo generaliza mejor a frases raras, code-switching ES/PT, y casos que no aparecen en training.

---

## 8. Deploy a HuggingFace Spaces

### Docker SDK vs Streamlit SDK

El Space original se creó con SDK `docker` — un Dockerfile que instalaba PyTorch, dependencias, y copiaba la app. Luego migramos a SDK `streamlit` que es mucho más simple.

#### Configuración con Streamlit SDK

```yaml
# README.md (YAML frontmatter)
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
```

No necesita Dockerfile. HF maneja el entorno automáticamente: instala `requirements.txt` y ejecuta `streamlit run app.py`.

#### ¿Cuándo necesitás Docker SÍ o SÍ?

1. **Dependencias del sistema operativo**: `libxml2`, `ffmpeg`, `poppler-utils`, `graphviz`, drivers ODBC, binarios nativos
2. **Puerto personalizado o múltiples servicios**: Flask, FastAPI, Gradio custom, backend + frontend
3. **Precarga de modelos en BUILD time**: para que el contenedor arranque con los modelos ya en disco
4. **Control fino del entrypoint**: `CMD` con flags raros, scripts multi-paso
5. **Versión específica de Python**: el SDK streamlit siempre usa la última estable

#### Comparativa

| Aspecto | Docker SDK | Streamlit SDK |
|---------|-----------|---------------|
| **Tiempo de build** | 5-15 min | 30 seg - 2 min |
| **Caché de modelos** | En la imagen Docker (persiste entre restarts) | Se descarga cada vez que el container se reinicia |
| **Control del entorno** | Total (apt, binarios, servicios) | Limitado a Python puro |
| **Complejidad** | Alta (Dockerfile, HEALTHCHECK, EXPOSE) | Mínima (solo `app_file`) |
| **Debug** | Logs del build + logs del container | Logs más simples |
| **Portabilidad** | La imagen corre en cualquier lado | Solo funciona en HF Spaces |

#### El trade-off real

- **Streamlit SDK**: sacrificás cold start más lento (modelos se descargan al arrancar) a cambio de build instantáneo y cero mantenimiento.
- **Docker SDK**: sacrificás tiempo de desarrollo y build a cambio de control total y arranque más rápido (modelos cacheados en la imagen).

Para ConversaAI la decisión correcta fue Streamlit SDK porque:
- No necesitamos librerías del sistema
- Los modelos ya están en HF Hub, se descargan solos
- Preferimos builds rápidos a cold starts rápidos
- Menos código que mantener (chau Dockerfile)

#### Error corregido

El Dockerfile original tenía:
```dockerfile
CMD ["streamlit", "run", "dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Pero el archivo se renombró a `app.py`. Se corrigió el CMD y después se eliminó el Dockerfile al migrar a Streamlit SDK.

---

### Arquitectura para producción (2M mensajes/mes)

El HF Space actual es una demo. Para un entorno real con 2M mensajes/mes (~46 msg/min), el problema no es la capacidad de cómputo (XLM-R en CPU procesa ~100-300ms por mensaje), sino la **arquitectura**.

#### Problemas del HF Space en producción

```
             usuario
               │
               ▼
┌─────────────────────────────┐
│      HF Space (1 proc)      │
│  ┌───────────────────────┐  │
│  │ Streamlit (UI + ML)   │  │  ← TODO en el mismo proceso
│  │ app.py hace todo      │  │     (UI, modelo, lógica)
│  └───────────────────────┘  │
└─────────────────────────────┘
```

1. **Sin API**: no hay forma de que un bot, CRM o sistema externo consuma las clasificaciones
2. **Sin cola de procesamiento**: si llegan 1000 mensajes a la vez, se cae
3. **Sin persistencia**: los resultados se pierden al cerrar el dashboard
4. **Sin separación de concerns**: si la UI crashea, se pierde el procesamiento
5. **Sin escalado horizontal**: no podés replicar instancias

#### Arquitectura propuesta

```
                    ┌──────────────┐
                    │   Dashboard  │  ← Solo consulta datos
                    │  (Streamlit) │     (lectura, sin ML)
                    └──────┬───────┘
                           │
┌───────┐    ┌─────────────v──────────┐    ┌──────────┐
│ CRM / │    │      Base de Datos     │    │   API    │
│ Chat  │───►│ (PostgreSQL / BigQuery)│◄───│ FastAPI  │
│ Bot   │    └─────────────┬──────────┘    └────┬─────┘
└───────┘                  │                     │
                           │                     │
                    ┌──────v──────┐     ┌────────v────────┐
                    │  Worker ML  │     │  Worker ML      │
                    │  (Docker)   │     │  (Docker)       │
                    │  GPU opcional│    │  GPU opcional   │
                    └─────────────┘     └─────────────────┘
                           │
                    ┌──────v──────┐
                    │    Cola     │
                    │  (RabbitMQ  │
                    │   / Kafka)  │
                    └─────────────┘
```

#### Por qué Docker (o similar) es necesario en producción

| Necesidad | Sin Docker | Con Docker (K8s / ECS) |
|-----------|-----------|----------------------|
| Escalar horizontal | No podés replicar | `kubectl scale deployment ml-worker --replicas=10` |
| GPU | Atado a CPU de HF Space | `docker run --gpus all` |
| Modelo en memoria | Único punto de falla | Cada worker tiene su copia |
| Deploy sin downtime | Imposible (parás la app) | Rolling update: 0 downtime |
| Aislar recursos | Todo en un proceso | Límites de CPU/RAM por contenedor |

#### Componentes de la arquitectura

1. **API (FastAPI)**: Endpoint REST para que bots/CRMs envíen mensajes y consulten resultados
2. **Cola (RabbitMQ/Kafka)**: Buffer para manejar picos de tráfico sin perder mensajes
3. **Workers ML (Docker)**: Contenedores que escuchan la cola, cargan el modelo, clasifican y guardan en DB. Se replican según demanda.
4. **Base de datos (PostgreSQL/BigQuery)**: Almacena todas las clasificaciones con sus metadatos
5. **Dashboard (Streamlit)**: Solo lee de la DB, no ejecuta ML. Puede estar en HF Space sin problemas.

#### Alternativa: Serverless (Lambda)

Podría funcionar, pero con modelos de ~556MB los cold starts son de 10-20 segundos, inviables para una API síncrona. Docker con workers persistentes es la opción correcta acá.

---

## 9. Cómo funciona Docker (fundamentos)

### Imagen vs Contenedor

```
Dockerfile (receta)  →  docker build  →  Imagen (foto estática)
                                           │
                                    docker run │
                                               ▼
                                        Contenedor (proceso vivo)
```

- **Imagen**: archivo con todo: SO mínimo + Python + librerías + código. Es **inmutable**.
- **Contenedor**: proceso corriendo a partir de una imagen. Podés tener 10 contenedores de la misma imagen.

### Capas (layers)

```dockerfile
FROM python:3.11-slim        ← Capa 1: SO + Python (~120MB)
WORKDIR /app                  ← Capa 2: metadata
COPY requirements.txt .       ← Capa 3: archivo
RUN pip install -r req.txt    ← Capa 4: dependencias (~1.5GB)
COPY . .                      ← Capa 5: código (~50KB)
CMD ["streamlit", "run", ...] ← Capa 6: entrypoint
```

Cada instrucción crea una **capa**. Si no cambiás una capa, Docker la reusa de caché. Por eso si solo cambiás `app.py`, las capas 1-4 vienen de caché y solo se reconstruye la 5 y 6 → build en segundos.

### Aislamiento
Cada contenedor tiene su propio sistema de archivos, red, procesos, y límites de CPU/RAM.

### Puertos
```bash
docker run -p 8501:8501 conversaai
#        host:container
```
Lo que llegue al puerto 8501 del host se redirige al 8501 del contenedor.

### Volúmenes (persistencia)
Los contenedores son efímeros. Si los borrás, perdés todo. Los volúmenes son carpetas que sobreviven:
```bash
docker run -v /ruta/en/host:/ruta/en/contenedor conversaai
```

---

## 10. Evaluación de Docker para el MVP

### ¿Qué ganaríamos con Docker ahora?

| Necesidad | ¿La tenemos hoy? | ¿Docker ayuda? |
|-----------|-----------------|----------------|
| API para que bots/CRMs consuman clasificaciones | ❌ No hay | ✅ FastAPI en otro contenedor |
| Procesar en background sin bloquear UI | ❌ Todo en Streamlit | ✅ Worker separado |
| Persistencia de datos | ❌ Se pierde al cerrar | ✅ DB en contenedor |
| Escalar si hay picos | ❌ Un solo proceso | ✅ Múltiples workers |
| Precargar modelos (cold start rápido) | ❌ Descarga cada vez | ✅ Caché en la imagen |

### ¿Qué complicaría?

- **Infraestructura**: necesitás un server (VPS, EC2) para correr Docker. HF Space no corre Docker Compose.
- **DevOps**: mantenimiento de contenedores, variables de entorno, orquestación.
- **Costo**: servidor 24/7 vs HF Space gratis.
- **Complejidad**: de 1 archivo (`app.py`) pasás a multi-servicio con Docker Compose.

### Conclusión para MVP

**No conviene.** El MVP resuelve:

1. ✅ **Demostrar que los modelos funcionan** — Streamlit lo hace perfecto
2. ✅ **Mostrar métricas y visualizaciones** — Plotly + Streamlit
3. ✅ **Probar con datos reales** — Subís un CSV y listo
4. ✅ **Compartir con stakeholders** — HF Space URL pública

Docker agregaría valor CUANDO:
- Necesités que un bot/CRM consuma la API en vez de subir CSVs
- El volumen requiera procesamiento asincrónico
- Necesités mantener histórico en DB
- El cold start del HF Space sea un problema real

Primero validar el negocio, después escalar la arquitectura.

## 11. Estructura del Proyecto

```
ConversaAI/
├── app.py                          ← Dashboard Streamlit único (HF Space)
├── requirements.txt                ← Dependencias
├── Dockerfile                      ← Referencia (HF Space usa SDK streamlit)
├── data/
│   └── corpus_demo_bilingue.csv    ← 108 mensajes demo ES/PT
├── notebooks/
│   ├── sentiment_xlmr.ipynb.py     ← Fine-tuning sentimiento
│   └── intent_xlmr.ipynb.py        ← Fine-tuning intención
├── src/
│   ├── pipeline.py                 ← Orquestador ConversaAIPipeline
│   ├── data/processor.py           ← DataProcessor: normalización + tokenización
│   ├── models/
│   │   ├── sentiment.py            ← SentimentClassifier (XLM-R + fallback)
│   │   └── intent.py               ← IntentClassifier (XLM-R + fallback)
│   └── features/
│       ├── churn.py                ← ChurnScorer (fórmula única de churn)
│       └── business_rules.py       ← Taxonomía + reglas de negocio + categorize_churn
├── scripts/
│   └── pipeline_fallback.py        ← Pipeline rule-based offline (delega churn a ChurnScorer)
├── tests/
│   ├── test_processor.py           ← 7 tests — DataProcessor
│   ├── test_churn.py               ← 19 tests — ChurnScorer
│   ├── test_pipeline_fallback.py   ← 33 tests — FallbackPipeline
│   ├── test_business_rules.py      ← 22 tests — reglas de negocio
│   └── test_pipeline_batch.py      ← 3 tests — batch predict
├── dashboard-web/                  ← Dashboard estático (Vercel/Pages, otro compañero)
└── README.md
```

---

## 12. Flujo de app.py

```
1. Configuración de página + sidebar (fuente de datos + navegación)
2. load_sample_corpus() → lee el CSV demo (o subida de CSV/Excel)
3. run_analysis(data) → pipeline completo:
   a. pipeline.batch_predict() → sentimiento + intención + churn (en batch real)
   b. Reglas de negocio unificadas (apply_business_rules)
   c. Macro-intenciones (macro_intent)
   d. Riesgo de churn categorizado (categorize_churn)
4. Visualización con Plotly (métricas, distribuciones, histograma, alertas)
5. Navegación por páginas: 📊 Dashboard / 🔍 Explorar / 💡 Acciones
```

---

## 13. Modelos en HF Hub

| Modelo | Clases | Accuracy | Repo |
|--------|--------|----------|------|
| `Rosela/xlm-r-sentiment-espt` | Negative / Neutral / Positive | ~92.2% | [Link](https://huggingface.co/Rosela/xlm-r-sentiment-espt) |
| `Rosela/xlm-r-intent-espt` | 9 intenciones ES | ~97.85% | [Link](https://huggingface.co/Rosela/xlm-r-intent-espt) |

---

## 14. Entrenamiento de Modelos (Fine-tuning)

### El concepto de fine-tuning

```
xlm-roberta-base (pre-entrenado en ~2.5TB de texto multilingüe)
       │
       │  Congelás el conocimiento general de lenguaje
       │  y le enseñás una tarea específica
       ▼
   Fine-tuning con 6000-24000 mensajes de soporte
       │
       ▼
   Modelo especializado en sentimiento o intención ES/PT
```

XLM-R ya sabe gramática, sintaxis y semántica en 100 idiomas. El fine-tuning solo ajusta sus pesos para la tarea específica (clasificación). Por eso con relativamente pocos datos (6000-24000) se obtienen resultados excelentes.

### Pipeline completo de entrenamiento

```
Datos crudos (parquet) → Split estratificado 80/10/10 → Tokenizer → Modelo → Trainer → Evaluación → Push a Hub
```

### Paso a paso

#### 1. Carga de datos
Los datos están en formato **Parquet** (eficiente, columnar). Tres archivos:
- `train.parquet` (80%) — aprende los patrones
- `val.parquet` (10%) — ajusta hiperparámetros y early stopping
- `test.parquet` (10%) — evaluación final, NO se toca durante training

```python
# Si no hay splits, se dividen automáticamente con stratified split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
)
```

**Stratified split** = mantiene la misma proporción de clases en train/val/test. Si `cancelacion` es el 20% de los datos, será ~20% en cada split.

#### 2. Tokenizer (XLM-R usa SentencePiece)
```python
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",   # Rellena hasta 128 tokens
        truncation=True,        # Corta textos más largos que 128
        max_length=MAX_LENGTH,
    )
```

- XLM-R usa **Byte-Level BPE** (Byte Pair Encoding): divide palabras en subunidades según frecuencia
- Ejemplo: "cancelación" → `["cancel", "ación"]` (cada parte tiene un ID en el vocabulario)
- `MAX_LENGTH=128` porque las conversaciones de soporte no requieren contexto largo
- XLM-R no tiene `pad_token` por defecto; se usa `eos_token` como pad

#### 3. Arquitectura del modelo

**Sentimiento** (3 clases):
```python
MODEL_CHECKPOINT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# ← Parte de un modelo YA fine-tuneado para sentimiento multilingüe
id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
```

**Intención** (9 clases):
```python
MODEL_CHECKPOINT = "xlm-roberta-base"
# ← Parte del XLM-R base, sin fine-tuning previo
id2label = {0: "cancelacion", 1: "consulta_general", ..., 8: "seguimiento"}
```

Diferencia clave: sentimiento usa un checkpoint YA entrenado para sentimiento (transfer learning sobre transfer learning), mientras que intención arranca del XLM-R base.

#### 4. TrainingArguments (hiperparámetros)

```python
training_args = TrainingArguments(
    learning_rate=2e-5,              # Tasa de aprendizaje
    per_device_train_batch_size=16,  # Ejemplos por batch por GPU
    per_device_eval_batch_size=32,   # Evalúa más rápido (batch más grande)
    num_train_epochs=5,              # Máximo de épocas
    weight_decay=0.01,               # Regularización L2
    warmup_ratio=0.1,                # 10% del training con LR creciente
    fp16=True,                       # Precisión mixta (entrena en 16-bit, más rápido)
    label_smoothing_factor=0.1,      # Anti-overfitting
    eval_strategy="epoch",           # Evalúa al final de cada época
    save_strategy="epoch",
    load_best_model_at_end=True,     # Guarda el mejor modelo (no el último)
    metric_for_best_model="accuracy" # (o "f1_macro" para intención)
)
```

#### 5. Métricas de evaluación

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)  # Toma la clase con mayor probabilidad
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
```

| Métrica | Qué mide |
|---------|----------|
| **Accuracy** | % de predicciones correctas sobre el total |
| **F1 weighted** | F1 promedio ponderado por soporte de cada clase |
| **F1 macro** | F1 promedio simple (cada clase pesa igual) — **métrica principal para intención** |

Para intención se usa **macro F1** porque hay 9 clases y algunas pueden tener menos datos. Macro F1 asegura que todas las clases importan por igual, no solo las mayoritarias.

#### 6. Trainer + Early Stopping

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],  # SOLO val, NUNCA test
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,      # Si después de 2 épocas no mejora, para
            early_stopping_threshold=0.001, # Mejora mínima para considerar progreso
        )
    ],
)
```

El Trainer de HuggingFace abstrae todo el loop de entrenamiento: forward pass, backward pass, optimizer step, logging, evaluación, checkpointing. Con 3 líneas configurás todo.

#### 7. Evaluación en test set (UNA SOLA VEZ al final)

```python
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
```

**Regla de oro**: el test set se usa EXACTAMENTE UNA VEZ. Si evaluás en test y no te gusta el resultado, NO ajustás el modelo y evaluás de nuevo (eso sería hacer trampa = data leakage). Volvés a ajustar hiperparámetros en val, y recién cuando estás conforme, evalúas en test una última vez.

#### 8. Push a Hugging Face Hub

```python
trainer.push_to_hub()           # Sube el modelo (config.json + model.safetensors)
tokenizer.push_to_hub(HUB_MODEL_ID)  # Sube el tokenizer
```

El Hub versiona automáticamente. Cada push genera un commit. Después cualquiera puede cargar el modelo con:
```python
pipeline("text-classification", model="Rosela/xlm-r-intent-espt")
```

### Comparativa: Sentimiento vs Intención

| Aspecto | Sentimiento | Intención |
|---------|------------|-----------|
| **Checkpoint base** | `cardiffnlp/twitter-xlm-r...` (ya fine-tuneado) | `xlm-roberta-base` (pre-entrenado genérico) |
| **Clases** | 3 (NEGATIVE, NEUTRAL, POSITIVE) | 9 (cancelacion, queja, reembolso, etc.) |
| **Métrica principal** | Accuracy (> 0.95) | Macro F1 (> 0.85) |
| **Datos de entrenamiento** | amazon_reviews_multi ES/PT + corpus_ecommerce | Bitext dataset (27 intents → 9 clases) |
| **Target** | 95% accuracy | 85% macro F1 |

### Anti-overfitting: 4 técnicas usadas

| Técnica | Qué hace | Por qué funciona |
|---------|----------|-----------------|
| **Early stopping** | Detiene el entrenamiento cuando val loss deja de mejorar | Evita que el modelo memorice el train set |
| **Weight decay (L2)** | Penaliza pesos grandes en la loss function | Fuerza pesos pequeños, modelo más simple |
| **Label smoothing** | Las labels target no son 0/1 exactos sino 0.9/0.05/0.05 | Evita que el modelo tenga demasiada confianza |
| **Stratified split** | Mantiene proporción de clases en cada split | Asegura que val y test representan la distribución real |

### Hardware

- **Plataforma**: Kaggle (GPU T4x2, 16 GB VRAM)
- **Tiempo**: ~2-3 horas (sentimiento), ~3-4 horas (intención), 5 épocas máximas
- **Precisión mixta (fp16)**: usa 16-bit en vez de 32-bit → ~2x más rápido, mínima pérdida de precisión
- **Los modelos NO se pueden entrenar en CPU**: 278M de parámetros, una época tomaría horas

---

## 15. Evolución de los modelos de Intención

### Timeline

```
Bitext original (27 clases)
       │
       ▼
Rosela/xlmr-intent (23 clases)   ← Tu compañero, ~6000 msgs
       │
       ├──→ Rosela/xlmr-intent-6clases (experimento intermedio)
       │
       ▼
Rosela/xlm-r-intent-espt (9 clases)  ← Vos, ~24000 msgs ← USADO EN PRODUCCIÓN
```

### Los modelos en HF Hub

#### `Rosela/xlmr-intent` — 23 clases (Bitext original limpiado)

```
0:  registration_problems
1:  check_cancellation_fee
2:  newsletter_subscription
3:  change_order
4:  switch_account
5:  change_shipping_address
6:  check_invoice
7:  cancel_order
8:  check_refund_policy
9:  delete_account
10: track_order
11: get_refund
12: contact_customer_service
13: create_account
14: check_payment_methods
15: recover_password
16: complaint
17: edit_account
18: review
19: track_refund
20: delivery_options
21: place_order
22: payment_issue
```

Entrenado por tu compañero con ~6000 mensajes. Etiquetas en INGLÉS. Especificidad máxima pero clases minoritarias con pocos datos.

#### `Rosela/xlmr-intent-6clases` — 6 clases (experimento intermedio, privado)

```
0: cancelacion
1: consulta
2: facturacion
3: problema_tecnico
4: reclamo
5: resuelto
```

Modelo PRIVADO (no visible públicamente). Etiquetas en español pero muy pocas clases. Demasiado genérico — `consulta` y `problema_tecnico` abarcan demasiado. No se usa.

#### `Rosela/xlm-r-intent-espt` — 9 clases (final, en producción)

```
0: cancelacion
1: consulta_general
2: facturacion_pago
3: feedback
4: gestion_cuenta
5: modificacion_pedido
6: queja
7: reembolso
8: seguimiento
```

Balance entre granularidad y precisión. Etiquetas en español pero el modelo acepta ES + PT.

#### `Rosela/conversaai-xml-r-sentiment-espt` — Vacío

Este modelo está **VACÍO**. Solo tiene `.gitattributes` y `README.md`, sin `model.safetensors` ni `config.json`. Fue creado como placeholder pero nunca se le subieron los pesos. No sirve para nada.

### ¿Cuál es mejor y por qué?

| Modelo | Clases | Granularidad | Precisión (F1 macro) | Veredicto |
|--------|--------|-------------|---------------------|-----------|
| `xlmr-intent` | 23 | Muy fina | ~0.80 | ❌ Clases chicas sufren |
| `xlmr-intent-6clases` | 6 | Muy gruesa | ? | ❌ Demasiado genérico |
| `xlm-r-intent-espt` | 9 | Intermedia | ~0.97 | ✅ **Gana** |

**El de 23 clases** es tentador por la especificidad ("change_shipping_address" vs "modify_order") pero en la práctica las clases con pocos ejemplos tienen F1 bajo. Además las etiquetas están en inglés, lo que agrega una capa de traducción extra.

**El de 6 clases** agrupa demasiado. "consulta" y "problema_tecnico" son tan amplios que no aportan información útil para tomar decisiones de negocio. "resuelto" no es una intención, es un estado. Fue un experimento para ver si con menos clases se ganaba precisión, pero se perdió demasiada información.

**El de 9 clases** es el punto dulce:
- Suficientemente específico para tomar decisiones de negocio (queja ≠ reembolso ≠ cancelación)
- Suficientemente general para que cada clase tenga muchos ejemplos → F1 alto
- Taxonomía alineada al negocio pensada desde el churn (delete_account se captura con heurística)  

---

## 16. Datasets Originales

### Sentimiento (3 clases: negative, neutral, positive)

| Dataset | Idioma | Registros | Clases originales | Fuente |
|---------|--------|-----------|-------------------|--------|
| `amazon_reviews_multi` | ES + PT | ~100k | 1-5★ → negative(1-2), neutral(3), positive(4-5) | HuggingFace Datasets |
| `spanish_amazon_reviews` | ES | ~160k | positive, negative (binario) | Archivo local |
| `portuguese_ecommerce_reviews` (corpus_ecommerce) | PT | ~312k | positive, negative (binario) | Archivo local |

### Intención (27 → 9 clases)

| Dataset | Idioma | Registros | Clases | Fuente |
|---------|--------|-----------|--------|--------|
| Bitext customer support | EN (traducido a ES/PT) | ~27k | 27 originales → 9 finales | HuggingFace Datasets |

### Dataset sintético neutral (nota importante)

**Problema original:** `corpus_ecommerce` (PT) y `spanish_amazon_reviews` (ES) eran datasets **binarios** (positive/negative). Para el modelo de **3 clases** se necesitaban ejemplos neutrales, especialmente en portugués.

**Solución planteada:** Generar ejemplos neutrales sintéticos a partir de consultas factuales típicas de e-commerce: preguntas sobre plazos de entrega, métodos de pago, horarios de atención, disponibilidad de productos, etc.

**Estado real:** El código de generación no se conservó en el repositorio. Es probable que el dataset sintético:
- Se haya generado en un Colab independiente que no se persistió
- O **nunca se haya incorporado al training set final**

**¿Por qué el modelo funciona igual sin ellos?**
1. `amazon_reviews_multi` ya proveía ~11k ejemplos neutrales vía reseñas de 3★ en ES y PT
2. XLM-R tiene **cross-lingual transfer**: los neutrales en ES ayudan a clasificar neutrales en PT
3. Las conversaciones reales de soporte ya contienen consultas factuales neutrales (horarios, plazos, etc.)

Conclusión: el dataset sintético era un extra pensado para balancear, pero XLM-R fue suficientemente robusto sin él.

---

## 17. Streamlit Session State

### ¿Qué problema resuelve?

Cada interacción en Streamlit (click, slider, file upload) **recarga todo el script**. Sin session state, los datos procesados se pierden en cada rerun.

### Cómo se usa en ConversaAI

En `app.py`:

```python
# Inicialización (solo la primera vez)
if "results_df" not in st.session_state:
    st.session_state.results_df = None
```

```python
# Guardar después de procesar
df_results, model_status = run_analysis(data)
st.session_state.results_df = df_results
st.session_state.model_status = model_status
```

```python
# Leer en cualquier rerun
if st.session_state.results_df is not None:
    show_dashboard(st.session_state.results_df)
```

### Flujo completo

```
1er load: session_state.results_df = None
  ▼
Usuario elige "Corpus de ejemplo" o sube CSV
  ▼
run_analysis() → guarda en session_state.results_df
  ▼
Rerun automático → lee de session_state → muestra dashboard
  ▼
Usuario cambia filtros en sidebar → rerun → lee del mismo results_df
  ▼
NUNCA vuelve a procesar el CSV → instantáneo
```

### ¿Qué más se podría persistir?

- Filtros seleccionados (sentimiento, intención, rango de fechas)
- Upload de archivo en sí (para no tener que subirlo de nuevo)
- Modelo cargado (aunque eso va mejor en `@st.cache_resource`)

### Patrón general

```python
# 1. Inicializar
if 'key' not in st.session_state:
    st.session_state.key = default_value

# 2. Actualizar (en callback o después de acción)
st.session_state.key = new_value

# 3. Leer en cualquier parte
value = st.session_state.key
```

Sin session state, Streamlit es un juguete. Con session state, es una app.

---

## 18. Dashboard Web (Vercel / GitHub Pages)

### ¿De qué se trata?

Dashboard **estático** HTML/CSS/JS hecho por otro compañero del equipo. No es el dashboard principal (ese es el de Streamlit), sino una **vista pública** del proyecto.

### Stack

- **HTML semántico** con tema dark/light toggle
- **CSS vanilla** separado por responsabilidades (SRP):
  - `main.css` — variables, reset, tipografía
  - `layout.css` — sidebar + shell + grid
  - `components.css` — tarjetas, botones, tabs
  - `charts.css` — contenedores de gráficos
- **JavaScript vanilla** con patrón Factory para gráficos:
  - `ChartFactory.js` — crea el tipo de gráfico según el nombre
  - `BarChart.js`, `DoughnutChart.js`, `LineChart.js`, `RadarChart.js` — cada uno en su archivo
  - `UIController.js` — maneja tabs, sidebar, tema
  - `app.js` — orquestador principal
- **Font Awesome** para íconos
- **Google Fonts** (Inter + Fira Code)
- **Sin framework** — cero dependencias JS

### Estructura

```
dashboard-web/
├── index.html              ← Entry point único (SPA)
└── assets/
    ├── css/
    │   ├── main.css        ← Variables, reset, tipografía
    │   ├── layout.css      ← Sidebar + shell responsive
    │   ├── components.css  ← Cards, botones, tabs
    │   └── charts.css      ← Contenedores de gráficos
    └── js/
        ├── app.js          ← Orquestador
        ├── charts/
        │   ├── ChartFactory.js  ← Factory pattern
        │   ├── BarChart.js
        │   ├── DoughnutChart.js
        │   ├── LineChart.js
        │   └── RadarChart.js
        └── ui/
            └── UIController.js  ← Tabs, sidebar, tema
```

### Tabs del dashboard

| Tab | Qué muestra |
|-----|-------------|
| **Resumen General** | KPIs globales, distribución de sentimiento e intención |
| **Comparativa de Modelos** | Accuracy, F1, precisión por modelo |
| **Análisis de Frustración** | Niveles de frustración por conversación |
| **Recomendaciones** | Insights y sugerencias basadas en datos |

### Deploy

Se deploya a **GitHub Pages** via GitHub Actions:

```yaml
# .github/workflows/pages.yml
on:
  push:
    branches: [main]
    paths:
      - dashboard-web/**

jobs:
  build:
    steps:
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dashboard-web
  deploy:
    needs: build
    steps:
      - uses: actions/deploy-pages@v4
```

También se hostea en **Vercel** (el compañero lo conectó manualmente):  
👉 `https://s04-26-equipo-40-data-science.vercel.app`

### Diferencia con el dashboard de Streamlit

| Aspecto | Streamlit (app.py) | Vercel (dashboard-web) |
|---------|-------------------|----------------------|
| **Creado por** | Vos | Otro compañero |
| **Tipo** | App dinámica (Python) | Sitio estático (HTML/JS) |
| **Hosting** | HuggingFace Spaces | GitHub Pages + Vercel |
| **Datos** | Procesa CSVs en vivo | Mock/data estático precargado |
| **Modelos** | Carga XLM-R desde HF Hub | No carga modelos (solo visualización) |
| **Propósito** | Análisis interactivo | Landing / showcase del proyecto |

---

## 19. Fallback Pipeline

### ¿Qué es y por qué existe?

El **Fallback Pipeline** es un pipeline 100% rule-based que NO carga modelos de ML. Usa solo keywords, regex, y reglas heurísticas — cero dependencias de `transformers`, `torch`, ni descarga de modelos. Se creó para el Demoday porque los modelos XLM-R fine-tuneados no estaban listos a tiempo y se necesitaba algo que funcione **siempre**, sin conexión a internet ni GPU.

```
FallbackPipeline (pipeline_fallback.py)
  100% rule-based, 0 dependencias
  Solo re (stdlib de Python)

  predict_sentiment() → keywords ES/PT + detección de negación
  predict_intent()    → regex patterns ES/PT (9 intents) + probabilidades
  compute_churn()     → delega a ChurnScorer (fórmula única del proyecto)
```

### Dos implementaciones independientes

| Aspecto | `FallbackPipeline` (`scripts/`) | `ConversaAIPipeline` (`src/`) |
|---------|--------------------------------|------------------------------|
| **Enfoque** | Rule-based puro desde cero | ML con degradación graceful |
| **Sentiment** | Keywords ES/PT c/ negación | XLM-R → `nlptown/bert-base-multilingual-uncased-sentiment` |
| **Intent** | Regex patterns | XLM-R → `MoritzLaurer/DeBERTa-v3-base-mnli` (zero-shot) |
| **Churn** | `ChurnScorer` (compartido) | `ChurnScorer` (misma clase) |
| **Dependencias** | Solo `re` + `ChurnScorer` | transformers, torch, modelos HF |
| **Velocidad** | Microsegundos por mensaje | 100-300ms por mensaje |
| **Precisión** | Baja/Media | Alta (~97-99%) |

Son **dos pipelines separados** para sentimiento e intención (uno rule-based, otro ML), pero **comparten la MISMA implementación de churn**: `ChurnScorer` (`src/features/churn.py`). Antes cada uno tenía su propia fórmula de churn con pesos distintos; hoy es una sola.

### Cómo funciona `FallbackPipeline` paso a paso

#### 1. Detección de idioma

```python
def detect_language(text: str) -> str:
    pt_markers = ["ão", "çã", "õe", "você", "vc", "obrigad", ...]
    score = sum(1 for m in pt_markers if m in text_lower)
    return "pt" if score >= 2 else "es"
```

Umbral: **2+ marcadores** de portugués en el texto → PT, sino ES.

#### 2. Sentiment por keywords

Tres diccionarios bilingües ES/PT (~25-30 keywords cada uno):

| Clase | Keywords ES | Keywords PT |
|-------|------------|-------------|
| `positive` | excelente, genial, bueno, me encantó, gracias... | excelente, ótimo, adorei, obrigado... |
| `negative` | pésimo, terrible, no funciona, estafa, queja... | péssimo, horrível, não funciona, golpe... |
| `neutral` | consulta, precio, horario, dónde está... | consulta, preço, horário, onde fica... |

**Reglas:**
1. **Negación**: si antes de 40 caracteres de la keyword hay un negador (`no`, `nunca`, `tampoco`, `jamás` en ES; `não`, `nunca`, `tampouco`, `jamais` en PT), la keyword NO cuenta
2. Se cuentan keywords de cada clase después de filtrar negaciones
3. Probabilidad = `count_clase / total_keywords_encontradas`
4. **Default** (sin keywords): `neutral` con prob 0.6

#### 3. Intent por regex

9 intents con patrones regex bilingües:

| Intent | Patrón ES ejemplo | Patrón PT ejemplo |
|--------|------------------|------------------|
| `cancelacion` | `cancel.a`, `anul.a` | `cancelamento`, `baixa` |
| `consulta_general` | `consulta`, `quiero saber` | `pergunta`, `quero saber` |
| `facturacion_pago` | `factur`, `pago`, `boleto` | `fatura`, `pagamento`, `boleto` |
| `feedback` | `feedback`, `me gust`, `excelente` | `feedback`, `gostei`, `adorei` |
| `gestion_cuenta` | `cuenta`, `contraseñ`, `registr` | `conta`, `senha`, `registr` |
| `modificacion_pedido` | `modificar mi pedido`, `compro` | `alterar meu pedido`, `comprei` |
| `queja` | `queja`, `pésimo`, `estafa` | `queixa`, `péssimo`, `golpe` |
| `reembolso` | `reembols`, `devolver`, `garantía` | `reembols`, `devolver`, `estorno` |
| `seguimiento` | `seguimient`, `tracking`, `rastre` | `acompanhament`, `tracking`, `rastrei` |

**Regla:** cada regex que matchea suma 1.0 al score. Gana el intent con mayor score. Si hay empate o score 0 → `consulta_general` con prob 1.0. Las probabilidades se normalizan sobre las 9 clases (`score / total`), con la probabilidad del mejor capada a 0.95.

#### 4. Churn scoring

El churn **NO se recalcula** en el fallback: `compute_churn()` delega a `ChurnScorer.score()`, la misma fórmula del pipeline principal (`w1=0.3, w2=0.35, w3=0.35`). Para eso, `predict_intent()` ahora devuelve también un dict de `probabilities` sobre las 9 clases, que es lo que `ChurnScorer` necesita.

### Churn: una sola fórmula (unificado)

Antes había **dos fórmulas de churn distintas** (`FallbackPipeline` usaba `0.5/0.15/0.2` y `ChurnScorer` usaba `0.3/0.35/0.35`), lo que hacía que el mismo mensaje diera números distintos según el pipeline. En el refactor se unificó: **ambos usan `ChurnScorer`**.

| Parámetro | Peso (único) |
|-----------|--------------|
| Sentimiento negativo | 0.3 |
| Frustración (keywords) | 0.35 |
| Intención de riesgo | 0.35 |

`ChurnScorer` además tiene `score_conversation()` con estrategia **worst-message-wins** (toma el mensaje más riesgoso de toda la conversación).

### app.py: el pipeline real

El `app.py` del dashboard usa **`ConversaAIPipeline`** (`src/pipeline.py`) — el mismo orquestador que une los clasificadores XLM-R + `ChurnScorer`. Después aplica las reglas de negocio unificadas (`apply_business_rules`) y categoriza el churn (`categorize_churn`). El pipeline tiene **modo fallback** integrado: si los modelos fine-tuneados no cargan, degrada gracefulmente a los fallbacks en vez de detener la app.

### ¿Cuándo usar cada uno?

| Escenario | Pipeline |
|-----------|----------|
| Dashboard demo con modelos fine-tuneados disponibles | `ConversaAIPipeline` (usado por `app.py`) |
| Entorno sin GPU, sin conexión, o sin transformers | `FallbackPipeline` |
| Sistema productivo con degradación graceful | `ConversaAIPipeline` |
| Necesitás churn por conversación (no por mensaje) | `ConversaAIPipeline` + `ChurnScorer.score_conversation()` |

La regla de oro: **el FallbackPipeline es para cuando NO hay ML disponible**. Si los modelos existen, siempre da mejores resultados usar XLM-R. El fallback rule-based es un "modo seguro" para que la app nunca se rompa.

---

## 20. CI/CD — Continuous Integration

**¿Qué es CI?**

CI = **Continuous Integration** (Integración Continua). Es una práctica donde cada cambio que subís a GitHub se **construye y testea automáticamente** en una máquina limpia. Si los tests pasan → ✅ verde. Si fallan → ❌ rojo, y no deberías mergear.

**El workflow `.github/workflows/ci.yml`:**

```yaml
on:
  push:
    branches: [main]       # Cada push a main
  pull_request:
    branches: [main]       # Cada PR contra main
```

Cuando eso se dispara, GitHub Actions:
1. Crea una VM **Ubuntu** limpia
2. Hace `checkout` del código
3. Setup **Python 3.11**
4. Instala dependencias: `pip install -r requirements.txt`
5. Corre todos los tests: `python -m pytest tests/ -v`

**¿Por qué es importante para portafolio?**

| Razón | Explicación |
|-------|-------------|
| **Estándar industrial** | TODA empresa seria bloquea PRs que rompen CI. No es opcional. |
| **Confianza** | Un badge verde en el README dice "este código funciona" sin que nadie tenga que clonar y probar. |
| **Detección temprana** | Si alguien pushea algo que rompe un test, CI avisa al INSTANTE, no 2 semanas después. |
| **Reproducibilidad** | Corre en entorno limpio, no "en mi máquina funciona". |
| **Profesionalismo** | Diferencia un proyecto tirado en GitHub de un repo SERIO. |

**El badge en el README:**

```markdown
[![CI](https://github.com/tu-usuario/tu-repo/actions/workflows/ci.yml/badge.svg)](...)
```

Se actualiza solo. Si está verde, el repo está saludable. Si está rojo, hay que arreglarlo antes de seguir.

---

## 21. Tests — La Red de Seguridad

Los tests no son burocracia. Son **lo que te permite cambiar código sin miedo**. Si rompés algo, los tests te avisan al instante en vez de descubrirlo en producción.

### Estructura

```
tests/
├── conftest.py              # Configuración global (agrega raíz al path)
├── test_processor.py        # DataProcessor.normalize_text() — 7 tests
├── test_churn.py            # ChurnScorer heurístico — 19 tests
├── test_pipeline_fallback.py # FallbackPipeline completo — 33 tests
├── test_business_rules.py   # Reglas de negocio + categorize_churn — 22 tests
└── test_pipeline_batch.py   # batch_predict (inferencia en batch) — 3 tests
```

**Total: 84 tests.**

---

### `conftest.py` — El pegamento

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
```

Agrega la raíz del proyecto al `sys.path` para que los tests puedan hacer `from src.data.processor import DataProcessor` sin romperse. Sin esto, Python no sabe dónde está `src/`.

---

### `test_processor.py` — 7 tests, 0 dependencias

Testea **DataProcessor.normalize_text()**, la función que limpia texto antes de que entre al pipeline. Si el normalizador falla, el pipeline entero recibe basura. Son tests **determinísticos**: sin GPU, sin modelos, corren en milisegundos.

| Test | ¿Qué prueba? | ¿Por qué importa? |
|------|-------------|-------------------|
| `test_normalize_text_removes_extra_spaces` | `"  HOLA   MUNDO  "` → `"HOLA MUNDO"` | Espacios extras crean tokens vacíos |
| `test_normalize_text_handles_unicode` | Acentos y diéresis se conservan | ES/PT tienen acentos — romperlos es fatal |
| `test_normalize_text_empty` | `""` → `""` | El pipeline no puede explotar con texto vacío |
| `test_normalize_text_whitespace_only` | `"   "` → `""` | Ídem, caso borde |
| `test_normalize_text_preserves_punctuation` | `¿?¡!` se conservan | En español los signos de pregunta importan |
| `test_normalize_text_newlines_to_spaces` | `\n` → espacios | Los modelos esperan texto plano |
| `test_normalize_text_tabs_to_spaces` | `\t` → espacios | El copypaste a veces pega tabs |

---

### `test_churn.py` — 19 tests

Testea la **heurística de churn**: reglas de negocio que calculan riesgo de que un cliente se vaya. Componentes:

**Pesos configurables:**
- `test_default_weights` — Verifica pesos por defecto (w1=0.3, w2=0.35, w3=0.35)
- `test_set_weights` — Verifica que se puedan cambiar

**Casos extremos de scoring:**
- `test_score_high_churn` → "Desisto, quiero cancelar mi cuenta" con sentimiento negativo. Espera `aggregate_score ≥ 0.7`. Si alguien toca los pesos y esto baja a 0.3, el test falla.
- `test_score_low_churn` → "Excelente servicio" con sentimiento positivo. Espera `score ≤ 0.2`. Asegura que mensajes felices no sean falsos positivos.

**Estrategia de conversación:**
- `test_conversation_worst_message_wins` — Si en una conversación de 3 mensajes uno tiene score 0.9, el resultado final ES 0.9 (el peor mensaje define el riesgo). Decisión de negocio importante.
- `test_conversation_empty` — Conversación vacía → score 0.0

**Frustración por idioma:**
- `test_frustration_keywords_es` — "pésimo" en español dispara frustración
- `test_frustration_keywords_pt` — "péssimo" en portugués dispara frustración
- `test_no_frustration_false_positive` — "Todo está perfecto, gracias" NO debe dar frustración. Evita falsos positivos.

**Detección parametrizada:**
```python
@pytest.mark.parametrize("text,expected", [
    ("queja", 1.0),
    ("reclamo", 1.0),
    ("não serve", 1.0),
    ("felicidades", 0.0),
    ("gracias", 0.0),
])
```
5 casos en una línea. Si alguien agrega "felicitaciones" como keyword de frustración por error, este lo detecta.

**Edge cases:**
- `test_score_with_python_float_edge_cases` — Sentimiento negativo + intención riesgo, todo con prob 1.0. Verifica que aggregate esté entre 0 y 1. Sin este test, un día alguien cambia el cálculo y suma 2.5 sin caparlo.

---

### `test_pipeline_fallback.py` — 33 tests

El más grande. Testea el **FallbackPipeline** (rule-based, sin GPU). Organizado en clases:

**`TestDetectLanguage` (4 tests):**
- ES: "hola buenos dias" → `"es"`
- PT: "não obrigado" → `"pt"`
- Múltiples marcadores PT: "péssimo atendimento, não funciona" → `"pt"`
- Vacío: `""` → `"es"` (default seguro)

**`TestSentiment` (8 tests):**
- Un test por combinación: positivo/negativo/neutral en ES y PT
- `test_empty_text` → no crashea, devuelve neutral con prob 1.0
- `test_negation_flips_sentiment` → "Pésimo servicio, **no es bueno**": la negación "no" refuerza el negativo, no lo revierte
- `test_has_probabilities` → el output tiene la estructura correcta (dict con `probabilities` anidado)
- `test_sentiment_keywords_not_empty` — los diccionarios de keywords tienen datos

**`TestIntent` (13 tests):**
- **9 intenciones en español**: cancelacion, queja, consulta_general, feedback, reembolso, facturacion_pago, gestion_cuenta, seguimiento, modificacion_pedido
- **2 intenciones en portugués**: cancelacion, queja
- `test_empty_text` → vacío va a consulta_general (default)
- `test_probability_range` → prob entre 0 y 1
- `test_intent_patterns_defined` → verifica que los 9 intents tengan patrones. Si agregás un intent nuevo y olvidás los patrones, falla.

**`TestChurn` (2 tests):**
- `test_high_churn` — Pipeline completo: predice sentimiento → intención → churn en flujo real
- `test_low_churn` — Ídem con mensaje feliz

**`TestPipeline` (3 tests):**
- `test_predict_full_output` — `predict()` devuelve dict con `text`, `sentiment`, `intent`, `churn`
- `test_batch_predict` — 3 textos de entrada → 3 resultados
- `test_get_model_status` — Siempre `{"sentiment": "fallback", "intent": "fallback"}`

**`TestModificacionPedido` (2 tests):**
- ES y PT — el intent con menos patrones, fácil de romper

---

### ¿Por qué 84 y no 5?

Cada test cubre **un caso específico**. No es repetir lo mismo:

```
5 tests genéricos  → "uh, parece que funciona"
84 tests específicos → "esto NO está roto en 84 escenarios distintos"
```

En producción, el caso que rompe todo es **siempre el que no testeaste**: el string vacío, los tabs, el portugués con acentos, la negación, la conversación de 3 mensajes donde uno es crítico, las reglas de negocio, el orden en batch predict. Los 84 tests cubren esos bordes.

**Cómo correrlos:**
```bash
pytest tests/ -v              # Todos, con detalle
pytest tests/test_churn.py    # Solo churn
pytest tests/ -k "sentiment"  # Solo tests con "sentiment" en el nombre
```

---

## 22. GitHub Actions en Profundidad

GitHub Actions es el **asistente que trabaja gratis en tu repo**. Te da una máquina virtual cada vez que pasa algo en tu repo (push, PR, etc.) y ahí ejecutás lo que quieras.

### Arquitectura: Eventos → Workflows → Jobs → Steps

```
[Evento] → dispara un [Workflow] → que tiene [Jobs] → cada Job tiene [Steps]
```

**Evento** = algo que pasa en GitHub:
- `push` a main
- `pull_request` abierto
- `issues.opened` (cuando alguien crea un issue)
- `workflow_dispatch` (lo disparás MANUAL desde la UI)
- `schedule` (cron, tipo "corré esto todos los días a las 8am")

**Workflow** = un archivo `.yml` en `.github/workflows/`. Cada archivo = un workflow distinto.

**Job** = una unidad de trabajo. Los jobs pueden correr en paralelo o en secuencia. Cada job tiene un `runs-on:` que dice qué SO usar.

**Step** = un comando individual dentro de un job. Podés usar `actions/checkout@v4` (baja el código) o cualquier comando bash con `run:`.

### Anatomy de un workflow

```yaml
name: CI                              # Nombre visible en GitHub

on:                                   # Eventos que lo disparan
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:                               # Nombre del job
    runs-on: ubuntu-latest            # SO de la VM
    
    steps:
      - uses: actions/checkout@v4     # Baja tu código
      
      - uses: actions/setup-python@v5 # Instala Python
        with:
          python-version: "3.11"
      
      - name: Install dependencies    # Paso con nombre
        run: pip install -r requirements.txt
      
      - name: Run tests
        run: python -m pytest tests/ -v
```

**¿Qué hace GitHub cuando hacés push?**

1. Ve el evento `push` en `main`
2. Busca todos los `.yml` en `.github/workflows/`
3. Encuentra este workflow, ve que el evento matchea
4. Crea una VM **Ubuntu limpia** — nada instalado, como una compu recién formateada
5. Ejecuta cada step en orden
6. Si un step falla → todo el workflow falla → ❌ rojo
7. Si todos pasan → ✅ verde

### Cosas que no se ven a simple vista

**Cada workflow corre en su propia máquina.** No comparte nada con otros workflows, no comparte archivos, no comparte variables de entorno. Es una VM efímera que se destruye al terminar.

**`actions/*` son acciones reutilizables.** La comunidad publica acciones como si fueran librerías:

| Acción | Qué hace |
|--------|----------|
| `actions/checkout@v4` | Baja tu código a la VM |
| `actions/setup-python@v5` | Instala la versión de Python que digas |
| `actions/upload-artifact@v4` | Guarda archivos generados (logs, builds) |
| `actions/cache@v4` | Cachea dependencias para acelerar builds |

**Cuando NO usás una action**, usás `run:` que es bash directo. Todo lo que se pueda hacer en terminal se puede acá.

### ¿Dónde ves los resultados?

En GitHub, arriba del repo hay una pestaña **Actions**:

```
Code  Issues  Pull requests  Actions  Projects  ...
```

Ahí ves el historial de todas las ejecuciones. Cada push genera una nueva entrada con:
- ✅ Verde = pasó
- ❌ Rojo = falló (con logs de qué step y por qué)
- 🟡 Amarillo = está corriendo

Si un test falla, entrás al workflow, ves el step que falló, y los logs completos.

### ¿Qué más se puede hacer además de tests?

| Caso | Ejemplo en el workflow |
|------|------------------------|
| Deploy automático | Cada push a main → build → deploy a producción |
| Linting | Correr ruff (Python) o ESLint (JS) en cada PR |
| Publicar a PyPI | Cuando creás un release, publicar automáticamente |
| Build de Docker | Construir y pushear la imagen a Docker Hub |
| Notificaciones | Mandar Slack cuando un PR está listo |

### Lo que importa para tu portfolio

No necesitás ser experto. El 80% de los workflows del mundo son casi idénticos al `ci.yml`. La diferencia entre alguien que "oyó hablar" y alguien que lo usó es:

1. **Saber que existe** `.github/workflows/`
2. **Entender** `on:` (eventos), `runs-on:` (SO), `steps:` (comandos)
3. **Saber leer y modificar** un `.yml` de CI

---

## 23. Fine-Tuning de XLM-R

**Modelo base:** `xlm-roberta-base` — 279M parámetros, multilingual (100 idiomas), pre-entrenado por Facebook. Es un transformer tipo BERT pero entrenado con 100 idiomas simultáneamente, lo que lo hace ideal para ES/PT.

**Qué hicimos:**
- Cargamos `xlm-roberta-base` y lo fine-tuneamos con datos propios
- **2 modelos separados:** sentimiento (3 clases: positive, neutral, negative) e intención (9 clases)
- Dataset bilingüe ES/PT con ~5k-10k mensajes etiquetados

**Hiperparámetros típicos:**
| Parámetro | Valor | ¿Por qué? |
|-----------|-------|-----------|
| Learning rate | `2e-5` | El estándar para fine-tuning de transformers. Más alto y desestabiliza los pesos pre-entrenados. |
| Batch size | 16 | Límite de VRAM en una T4. Más grande = más estable, pero no entra. |
| Épocas | 3-5 | Más de 10 es overfitting casi seguro. El modelo ya sabe lenguaje, solo necesita adaptarse. |
| Optimizer | AdamW | Adam con weight decay. El estándar de la industria desde 2019. |
| Scheduler | Linear warmup | Calienta el LR gradualmente las primeras steps, luego lo reduce linealmente. |

**Métricas reportadas:**
- Sentimiento: ~96% accuracy
- Intención: ~97.85% accuracy

**Interpretación crítica (esto te diferencia en entrevistas):**
97% en intención suena increíble, PERO si `consulta_general` es el 60% del dataset, un modelo que siempre prediga `consulta_general` ya tiene 60% sin aprender NADA. El accuracy global miente cuando hay clases desbalanceadas. **Siempre pedir:**
- Matriz de confusión (qué clases confunde con cuáles)
- F1-score por clase (macro avg, no weighted)
- Support (cuántos ejemplos tiene cada clase)

**GPU:** Una sola T4 de Colab (~15GB VRAM). Tiempo: 30-60 min por modelo.

**Los pesos NO están en el repo.** Están en HuggingFace Hub:
- `Rosela/xlm-r-sentiment-espt` (~1GB)
- `Rosela/xlm-r-intent-espt` (~1GB)
- Se cargan con `AutoModelForSequenceClassification.from_pretrained()` (dentro de `SentimentClassifier`/`IntentClassifier`, vía `ConversaAIPipeline`)

---

## 24. Pipeline de Datos — El Viaje Completo

```
CSV crudo → reparar_texto() → ConversaAIPipeline.batch_predict() (sentimiento + intención + churn) → apply_business_rules() → categorize_churn() → Dashboard
```

**Paso a paso real en `app.py`:**

| Paso | Función | Entrada → Salida | ¿Qué hace? |
|------|---------|-----------------|------------|
| 1 | `pd.read_csv()` | archivo.csv → DataFrame | Carga los datos |
| 2 | `reparar_texto()` | "Ã±" → "ñ" | Corrige encoding roto. Sin esto los acentos son basura. |
| 3 | `normalize_text()` | "HOLA  MUNDO\n" → "HOLA MUNDO" | Limpia espacios extra, tabs, newlines (dentro del pipeline) |
| 4 | `batch_predict()` | lista de textos → sentimiento + intención + churn | Inferencia XLM-R **en batch real** (un forward por lote, no por mensaje) |
| 5 | `apply_business_rules()` | (texto, intent, sentimiento) → corregidos | Corrige falsos positivos del modelo |
| 6 | `categorize_churn()` | score 0-1 → "Alto/Medio/Bajo Riesgo" | Umbrales de negocio sobre la fórmula única |
| 7 | Dashboard | DataFrame → gráficos | Streamlit + Plotly |

**Las 6 reglas de negocio post-pipeline** (`apply_business_rules` en `src/features/business_rules.py`):
1. **Eliminación de cuenta** → override a `delete_account` (intención sintética, siempre Alto Riesgo)
2. **Cancelación de suscripción** → override a `cancelacion`
3. **Modificación de pedido** (cambiar talla/color/pedido) → override a `modificacion_pedido` (y suaviza sentimiento negativo a neutral)
4. **Positive + intención crítica** (cancelacion, queja, reembolso, facturacion_pago) → override a `feedback` (el modelo confunde quejas educadas con cancelaciones)
5. **consulta_general + keywords de queja** ("pésimo", "estafa") → override a `queja` (y fuerza sentimiento negativo)
6. **Positive + consulta_general + keywords de feedback** ("gracias", "excelente") → override a `feedback`

**Arquitectura de pipeline real:** cada etapa recibe datos de la anterior, los transforma, y los pasa a la siguiente. Si el modelo fine-tuneado no carga, el pipeline **degrada gracefulmente** al fallback (no corta la app con `st.stop()`).

---

## 25. Docker

**¿Qué es?** Una forma de empaquetar tu app + todo lo que necesita (Python, librerías, modelos) en un **contenedor** que corre igual en cualquier lado.

**Nuestro Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD streamlit run app.py
```

| Instrucción | ¿Qué hace? |
|-------------|------------|
| `FROM python:3.11-slim` | Imagen base con Python 3.11, versión minimizada (sin compiladores, sin tools de sistema) |
| `WORKDIR /app` | Crea `/app` como carpeta de trabajo |
| `COPY . .` | Copia TODO el código del repo al contenedor |
| `RUN pip install ...` | Instala dependencias |
| `EXPOSE 8501` | Puerto de Streamlit |
| `CMD streamlit run app.py` | Comando que se ejecuta al arrancar el contenedor |

**Comandos para usarlo:**
```bash
docker build -t conversaai .          # Construir la imagen
docker run -p 8501:8501 conversaai    # Correr el contenedor
# Después abrís http://localhost:8501
```

**Docker NO es una VM:**
- Una VM virtualiza hardware → corre un sistema operativo completo → pesa GB
- Docker comparte el kernel del host → aísla solo el espacio de usuario → liviano
- Se dice "Docker corre en **contenedores**, no en máquinas virtuales"

**¿Por qué existe si ya tenemos HF Spaces?**
- HF Spaces con SDK streamlit corre nuestra app, PERO atado a HF
- Docker es **portable**: lo levantás en AWS, GCP, Azure, Railway, tu propio server, donde sea
- El Dockerfile es el "plano" de tu infraestructura

**Problema típico con `slim`:** Si necesitás compilar algo nativo (ciertas versiones de torch, numpy), `slim` no tiene compilador. Ahí usás `python:3.11` (la versión completa, ~300MB más pesada).

---

## 26. Deployment a HuggingFace Spaces

**¿Qué es HF Spaces?** Es como Vercel/Netlify pero específico para apps de ML. Hosting gratuito con integración directa al Hub de modelos.

**Dos formas de deploy:**

| SDK streamlit | SDK docker |
|---------------|------------|
| HF crea el entorno automáticamente | Vos controlás TODO con tu Dockerfile |
| Build rápido (2-3 min) | Build lento (15+ min por torch) |
| Ideal para demos y portfolios | Para cuando necesitás algo no estándar |
| **Este es el que usamos ahora** | Estaba antes |

**Por qué cambiamos de docker a streamlit SDK:**
- Build con Docker: 15+ min (descargaba torch + modelos cada vez)
- Build con Streamlit SDK: 2-3 min (torch ya está pre-instalado, modelos se cachean)
- Diferencia enorme en la práctica

**Configuración en `README.md` del Space** (no confundir con el README del repo):
```
---
title: ConversaAI Dashboard
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
---
```

**Cómo se actualiza automáticamente:**
1. Conectás el Space a tu repo de GitHub (Settings → Git)
2. Cada push a `main` → HF detecta el cambio → rebuild
3. El badge en el README del repo muestra el estado

**Limitaciones que tenés que saber (para entrevistas):**
- **Cold start:** Si nadie visita el Space por un rato, la VM se duerme. La próxima visita tarda ~30 seg en arrancar.
- **Recursos limitados:** 2 vCPU, 16GB RAM (gratis). No corre modelos grandes (ni LLaMA, ni fine-tuning).
- **No es para producción real:** Es para demos, portfolios, prototipos.
- **Sin SSL personalizado:** Usás el subdominio de HF, no podés poner tu dominio.

---

## 27. Cloud Deployment (AWS / GCP / Azure)

HF Spaces está buenísimo para portfolio, pero en la industria real se deploya en la nube. Los 3 grandes son **AWS** (Amazon), **GCP** (Google), **Azure** (Microsoft). Son más un ecosistema que un solo servicio.

### Los servicios clave para ML

| Categoría | AWS | GCP | Azure |
|-----------|-----|-----|-------|
| Cómputo (VMs) | EC2 | Compute Engine | Virtual Machines |
| Contenedores | ECS / EKS | GKE | AKS |
| Serverless | Lambda | Cloud Functions | Functions |
| ML entrenamiento | SageMaker | Vertex AI | Azure ML |
| ML inferencia | SageMaker | Vertex AI | Azure ML |
| Base de datos | RDS | Cloud SQL | Azure SQL |
| Almacenamiento | S3 | Cloud Storage | Blob Storage |
| API Gateway | API Gateway | Cloud Endpoints | API Management |

**No necesitás saberlos todos.** Si mencionás **S3** (AWS), **GKE** (GCP), y decís que son equivalentes entre sí, ya mostrás que entendés el concepto.

### ¿Cómo deployarías ConversaAI en producción?

**Opción barata (como HF Spaces pero en serio):**
```
1. Docker build → push a Container Registry (ECR/GCR/ACR)
2. Correr el contenedor en Cloud Run (GCP) o App Runner (AWS)
3. Conectar a una base de datos manejada (RDS/Cloud SQL)
4. Ponerle un dominio custom con SSL
```

**Opción escalable (miles de requests por segundo):**
```
1. Modelo exportado a un endpoint dedicado (SageMaker / Vertex AI)
2. API en contenedores auto-escalables (ECS/GKE/AKS)
3. Load balancer adelante distribuyendo tráfico
4. Base de datos con réplicas de lectura
5. CDN para assets estáticos (CloudFront / Cloud CDN)
```

### ¿Cuál elegir?

| Factor | AWS | GCP | Azure |
|--------|-----|-----|-------|
| ML nativo | SageMaker (más maduro) | Vertex AI (excelente integración con TF) | Azure ML (mejor integración con Office/Teams) |
| Facilidad | Curva alta, muchísimos servicios | Más simple, UX más limpia | Buena si ya usás Microsoft |
| Precio | Caro, paga por todo | Competitivo, descuentos por uso continuo | Similar a AWS |
| Popularidad | ~60% del mercado | ~20% | ~20% |

**Regla práctica:** La empresa donde trabajes ya eligió uno. Tu tarea es entender los conceptos (VMs, contenedores, serverless, auto-escalado), no saberte todos los servicios de los 3.

### Conceptos que tenés que saber SÍ O SÍ

| Concepto | Explicación |
|----------|-------------|
| **Auto-escalado** | La app arranca más servidores cuando hay más tráfico y los apaga cuando baja. Como tener Netflix que en hora pico usa más servidores que a las 4am. |
| **Load Balancer** | Un "recepcionista" que distribuye requests entre múltiples servidores para que ninguno se sature. |
| **Región** | Dónde están físicamente los servidores (us-east-1, eu-west-1, sa-east-1). Importante para latencia y leyes de datos. |
| **Cold start** | El tiempo que tarda en arrancar un servidor que estaba apagado. En serverless (Lambda) puede ser 200ms-1s. |
| **SLA** | Service Level Agreement. Cuánto tiempo de actividad garantiza el proveedor. AWS: 99.99% = ~4 min de caída por mes. |
| **VPC / Red** | Tu red virtual aislada adentro de la nube. Los servidores no se exponen a internet directamente. |

### Para tu portfolio: HF Spaces está bien

No necesitás deployar en AWS para el portfolio. Lo que importa es que **sepás explicar**:
1. "Esto está en HF Spaces como demo"
2. "Si pasara a producción, lo metería en un contenedor Docker en Cloud Run"
3. "El modelo lo serviría con un endpoint dedicado para no bloquear la UI"

Eso muestra que entendés la diferencia entre demo y producción, que es lo que buscan en un ML Engineer Jr.

---

## Enlaces

| Recurso | URL |
|---------|-----|
| Dashboard (HF Space) | https://rosela-conversaai-dashboard.hf.space |
| Repositorio GitHub | https://github.com/No-Country-simulation/S04-26-Equipo-40-Data-Science |
| Perfil HF Hub | https://huggingface.co/Rosela |
