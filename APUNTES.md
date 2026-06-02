# 📚 Apuntes — ConversaAI

> Guía de estudio del pipeline de NLP, dashboard, y decisiones de arquitectura.
> Creada el 30/05/2026.

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

### Cálculo de riesgo de churn (`app.py`)

```python
INTENCIONES_CRITICAS = ['cancel_order', 'complaint', 'get_refund', 'payment_issue', 'delete_account']

def calcular_riesgo_churn(row):
    if row['Micro-Intención'] == 'delete_account':
        return 'Alto Riesgo'                    # 🔴 El usuario se quiere ir ya
    elif row['Sentimiento'] == 'Negative' and row['Micro-Intención'] in INTENCIONES_CRITICAS:
        return 'Alto Riesgo'                    # 🔴 Frustrado + intención crítica
    elif row['Sentimiento'] == 'Negative' or row['Micro-Intención'] == 'contact_customer_service':
        return 'Riesgo Medio'                   # 🟠 Algo no está bien
    return 'Riesgo Bajo'                        # 🟢 Happy path
```

Se convierte a número para promedios: `Alto=0.90`, `Medio=0.45`, `Bajo=0.10`.

**¿Por qué no un modelo de ML para churn?**
1. Faltan datos etiquetados de churn real (necesitarías conectar con datos de retención post-venta)
2. Las reglas son transparentes: cualquier stakeholder entiende por qué alguien está en Alto Riesgo

---

## 3. Regla de Modificación de Pedido

**Problema**: El modelo de ML no siempre detecta cuando un usuario quiere modificar un pedido (ej: "pedí la roja pero necesito la azul").

```python
if ('cambi' in texto_min or 'modific' in texto_min) and \
   ('talla' in texto_min or 'color' in texto_min or 'pedido' in texto_min):
    es_modificacion = True
elif 'pedí' in texto_min and 'necesito' in texto_min:
    es_modificacion = True
```

- Si se detecta modificación, baja el sentimiento a Neutral para no sesgar métricas
- Una modificación no es necesariamente una experiencia negativa

---

## 4. Regla de Eliminación de Cuenta

**Problema**: El modelo de 9 clases NO incluye `delete_account`. Sin esta regla, "quiero borrar mi cuenta" se clasifica como `gestion_cuenta` o `consulta_general`, perdiendo la señal de churn más crítica.

```python
palabras_abandono = [
    'excluir', 'apagar minha conta', 'cancelar conta',
    'borrar mi cuenta', 'cancelar mi cuenta', 'eliminar',
    'elimino mi cuenta', 'cerrar mi cuenta', 'dar de baja', 'borrar cuenta'
]
es_abandono = any(p in texto_min for p in palabras_abandono) and 'cuenta' in texto_min
```

Incluye palabras en español y portugués.

---

## 5. Mapeo de 9 Clases (INTENT_LABEL_MAP)

**¿Por qué existe?** El modelo `Rosela/xlm-r-intent-espt` predice etiquetas en español (ej: `queja`), pero el dashboard original se diseñó con labels en inglés. El mapa traduce:

| Label del modelo (ES) | Label del dashboard (EN) |
|----------------------|-------------------------|
| cancelacion | cancel_order |
| consulta_general | contact_customer_service |
| facturacion_pago | payment_issue |
| feedback | review |
| gestion_cuenta | edit_account |
| modificacion_pedido | modify_order |
| queja | complaint |
| reembolso | get_refund |
| seguimiento | track_order |

```python
df['Micro-Intención'] = [INTENT_LABEL_MAP.get(r['label'], 'consulta_general') for r in res_intent]
```

Si el modelo devuelve una etiqueta no mapeada, cae a `consulta_general` como default.

---

## 6. Soporte de Portugués

**¿El modelo funciona con portugués?** Sí. XLM-R es multilingüe y el modelo se entrenó con ES + PT (por eso el nombre termina en `-espt`). Acepta texto en portugués y etiqueta correctamente, pero las etiquetas de salida SIEMPRE están en español:

| Texto PT | Label que predice |
|----------|-------------------|
| "Quero cancelar meu pedido" | cancelacion |
| "Recebi o produto errado" | queja |
| "Como faço para trocar a cor?" | modificacion_pedido |

El `INTENT_LABEL_MAP` funciona igual para ambos idiomas. El idioma de cada mensaje se detecta aparte para la gráfica de evolución temporal.

**Problema de diseño**: Las labels solo están en español y se traducen con un mapa. Idealmente deberían ser bilingües consistentes (todo en inglés o códigos como `CANCEL_ORDER`).

---

## 7. Carga de Modelos con Caché

```python
@st.cache_resource 
def load_ai_models():
    intent_pipe = pipeline("text-classification", model="Rosela/xlm-r-intent-espt")
    sentiment_pipe = pipeline("text-classification", model="Rosela/xlm-r-sentiment-espt")
    return intent_pipe, sentiment_pipe

intent_clf, sentiment_clf = load_ai_models()
```

- `@st.cache_resource` es un decorador de Streamlit que hace que la función se ejecute **una sola vez** y cachee el resultado. No importa cuántas veces recargues la página o interactúes — los modelos se cargan UNA VEZ.
- Los modelos se descargan de HF Hub (~556MB c/u).
- `pipeline` de `transformers` hace TODO automáticamente:
  1. Descarga config.json, tokenizer, y pesos desde HF Hub
  2. Crea el tokenizer (XLM-R usa SentencePiece, un tokenizer basado en subword/byte-level BPE)
  3. Carga los pesos en la arquitectura correcta
  4. Devuelve un callable que hace: `tokenize → forward pass → postprocess`

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
├── app.py                          ← Dashboard Streamlit principal
├── requirements.txt                ← Dependencias
├── proyecto/
│   ├── Dockerfile                  ← Ya no se usa (referencia)
│   ├── data/
│   │   └── corpus_demo_bilingue.csv  ← 108 mensajes demo ES/PT
│   ├── notebooks/
│   │   ├── sentiment_xlmr.ipynb.py  ← Fine-tuning sentimiento
│   │   └── intent_xlmr.ipynb.py     ← Fine-tuning intención
│   └── src/
│       ├── pipeline.py              ← Orquestador
│       ├── data/processor.py        ← Normalización ES/PT
│       ├── models/sentiment.py      ← SentimentClassifier
│       ├── models/intent.py         ← IntentClassifier
│       └── features/churn.py        ← ChurnScorer heurístico
├── dashboard-streamlit/            ← Versión anterior (referencia)
└── README.md
```

---

## 12. Flujo de app.py

```
1. Configuración de página + carga de modelos (caché)
2. load_sample_corpus() → lee el CSV demo
3. procesar_dataframe(df) → pipeline completo:
   a. Limpieza (encoding repair: latin1/cp1252 → utf-8)
   b. Clasificación con modelos HF Hub
   c. Reglas de negocio (modify_order, delete_account)
   d. Macro-intenciones (agrupación en categorías de negocio)
   e. Riesgo de churn (heurístico)
4. mostrar_dashboard(df) → visualización con Plotly
5. Tabs: Dashboard / Métricas / Prioridades
```

---

## 13. Modelos en HF Hub

| Modelo | Clases | Accuracy | Repo |
|--------|--------|----------|------|
| `Rosela/xlm-r-sentiment-espt` | Negative / Neutral / Positive | ~99.9% | [Link](https://huggingface.co/Rosela/xlm-r-sentiment-espt) |
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
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
```

```python
# Guardar después de procesar
st.session_state.df_processed = procesar_dataframe(df, uploaded_file)
```

```python
# Leer en cualquier rerun
if st.session_state.df_processed is not None:
    mostrar_dashboard(st.session_state.df_processed)
```

### Flujo completo

```
1er load: session_state.df_processed = None
  ▼
Usuario elige "Demo 108 mensajes" o sube CSV
  ▼
procesar_dataframe() → guarda en session_state.df_processed
  ▼
Rerun automático → lee de session_state → muestra dashboard
  ▼
Usuario cambia filtros en sidebar → rerun → lee del mismo df_processed
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
  predict_intent()    → regex patterns ES/PT (9 intents)
  compute_churn()     → fórmula con pesos hardcodeados
```

### Dos implementaciones independientes

| Aspecto | `FallbackPipeline` (`scripts/`) | `ConversaAIPipeline` (`src/`) |
|---------|--------------------------------|------------------------------|
| **Enfoque** | Rule-based puro desde cero | ML con degradación graceful |
| **Sentiment** | Keywords ES/PT c/ negación | XLM-R → `nlptown/bert-base-multilingual-uncased-sentiment` |
| **Intent** | Regex patterns | XLM-R → `MoritzLaurer/DeBERTa-v3-base-mnli` (zero-shot) |
| **Churn** | Fórmula inline propia | `ChurnScorer` (clase separada, pesos configurables) |
| **Dependencias** | Solo `re` | transformers, torch, modelos HF |
| **Velocidad** | Microsegundos por mensaje | 100-300ms por mensaje |
| **Precisión** | Baja/Media | Alta (~97-99%) |

Son **dos pipelines completamente separados**, no es que uno caiga en el otro. Comparten el mismo `ChurnScorer` en la lógica, pero cada uno tiene su propia implementación de churn.

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

**Regla:** cada regex que matchea suma 1.0 al score. Gana el intent con mayor score. Si hay empate o score 0 → `consulta_general` con prob 0.3. Probabilidad final: `min(score / 3.0, 0.95)`.

#### 4. Churn scoring

Tres componentes con pesos fijos:

| Componente | Peso | Explicación |
|-----------|------|-------------|
| Sentimiento negativo | `prob_negative * 0.5` | A mayor negatividad, mayor riesgo |
| Frustración (keywords) | `min(count * 0.15, 0.3)` | Palabras como "urgente", "exijo", "no aguanto más" |
| Intención de riesgo | 0.2 si intent ∈ {cancelacion, queja, reembolso} | Ciertas intenciones son red flag |

```python
aggregate = min(sentiment_contrib + frustration_contrib + intent_contrib, 1.0)
```

### Churn: dos versiones distintas

El `ChurnScorer` de `ConversaAIPipeline` (en `churn.py`) tiene **pesos configurables** diferentes:

| Parámetro | FallbackPipeline (hardcodeado) | ChurnScorer (default) |
|-----------|-------------------------------|----------------------|
| Sentimiento negativo | 0.5 | 0.3 |
| Frustración keywords | 0.15 c/u, max 0.3 | 0.35 |
| Intención de riesgo | 0.2 | 0.35 |

Además `ChurnScorer` tiene `score_conversation()` con estrategia **worst-message-wins** (toma el mensaje más riesgoso de toda la conversación), mientras que `FallbackPipeline` solo analiza mensaje por mensaje.

### app.py: el pipeline real

El `app.py` del dashboard **NO usa** ninguna de las dos implementaciones anteriores. Carga los modelos XLM-R directamente con `pipeline()` de HuggingFace y aplica las reglas de negocio (modify_order, delete_account) inline. No tiene un "modo fallback" real — si los modelos no cargan, la app se detiene con `st.error()` + `st.stop()`.

### ¿Cuándo usar cada uno?

| Escenario | Pipeline |
|-----------|----------|
| Dashboard demo con modelos fine-tuneados disponibles | `app.py` (carga directa XLM-R) |
| Entorno sin GPU, sin conexión, o sin transformers | `FallbackPipeline` |
| Sistema productivo con degradación graceful | `ConversaAIPipeline` |
| Necesitás churn por conversación (no por mensaje) | `ConversaAIPipeline` + `ChurnScorer.score_conversation()` |

La regla de oro: **el FallbackPipeline es para cuando NO hay ML disponible**. Si los modelos existen, siempre da mejores resultados usar XLM-R. El fallback rule-based es un "modo seguro" para que la app nunca se rompa.

---

## Enlaces

| Recurso | URL |
|---------|-----|
| Dashboard (HF Space) | https://rosela-conversaai-dashboard.hf.space |
| Repositorio GitHub | https://github.com/No-Country-simulation/S04-26-Equipo-40-Data-Science |
| Perfil HF Hub | https://huggingface.co/Rosela |
