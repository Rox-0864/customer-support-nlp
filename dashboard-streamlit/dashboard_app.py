# ============================================
# CONVERSAI - DASHBOARD INTERACTIVO EN STREAMLIT
# ============================================
# Dashboard para el equipo de producto
# Basado en el nuevo pipeline: XLM-R (sentimiento 3 clases +
# intención 9 clases) + ChurnScorer heurístico
#
# Ejecutar: streamlit run dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple

# ── Plotting ─────────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go

# ── Importar el pipeline ─────────────────────────────────────────────────────
_PROYECTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proyecto")
if _PROYECTO_DIR not in sys.path:
    sys.path.insert(0, _PROYECTO_DIR)

try:
    from src.pipeline import ConversaAIPipeline
    _HAS_PIPELINE = True
except ImportError as e:
    _HAS_PIPELINE = False
    _PIPELINE_ERR = str(e)

# ===================================================================
# CONFIGURACIÓN DE PÁGINA
# ===================================================================
st.set_page_config(
    page_title="ConversaAI — Dashboard de Insights",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================================
# CONSTANTES
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "proyecto", "data")
SAMPLE_CSV = os.path.join(BASE_DIR, "..", "proyecto", "data", "corpus_demo_bilingue.csv")

INTENT_LABELS = [
    "cancelacion", "consulta_general", "facturacion_pago", "feedback",
    "gestion_cuenta", "modificacion_pedido", "queja", "reembolso", "seguimiento",
]

INTENT_DISPLAY = {
    "cancelacion": "Cancelación",
    "consulta_general": "Consulta General",
    "facturacion_pago": "Facturación/Pago",
    "feedback": "Feedback",
    "gestion_cuenta": "Gestión de Cuenta",
    "modificacion_pedido": "Modificación Pedido",
    "queja": "Queja",
    "reembolso": "Reembolso",
    "seguimiento": "Seguimiento",
}

SENTIMENT_COLORS = {
    "negative": "#e74c3c",
    "neutral": "#f39c12",
    "positive": "#2ecc71",
}

MACRO_INTENT_MAP = {
    "cancelacion": "Gestión de Pedidos",
    "modificacion_pedido": "Gestión de Pedidos",
    "seguimiento": "Gestión de Pedidos",
    "reembolso": "Pagos y Reembolsos",
    "facturacion_pago": "Pagos y Reembolsos",
    "gestion_cuenta": "Gestión de Cuenta",
    "queja": "Quejas y Reclamos",
    "feedback": "Feedback",
    "consulta_general": "Atención al Cliente",
}

DELETE_ACCOUNT_KW = [
    "excluir", "apagar minha conta", "cancelar conta",
    "borrar mi cuenta", "cancelar mi cuenta", "eliminar",
    "elimino mi cuenta", "cerrar mi cuenta", "dar de baja", "borrar cuenta",
]

CANCELACION_KW = [
    "cancel", "anul", "baja", "baixa", "desuscrib", "desinscrev",
    "no me interesa seguir", "no quiero seguir", "quiero darme de baja",
]

SUSCRIPCION_KW = [
    "suscrip", "assinatur", "subscription",
    "sócio", "socio", "plan", "plano",
    "membresía", "membresia", "servicio", "serviço",
    "contrato", "mensualidad", "cuota",
]

MODIFY_ORDER_RULES = [
    (("cambi", "modific"), ("talla", "color", "pedido")),
    (("pedí",), ("necesito",)),
]

# ===================================================================
# HELPER — Pipeline singleton (cached)
# ===================================================================
@st.cache_resource
def get_pipeline() -> Optional["ConversaAIPipeline"]:
    """Inicializa el pipeline ConversaAIPipeline (se cachea en memoria)."""
    if not _HAS_PIPELINE:
        return None
    try:
        return ConversaAIPipeline()
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el pipeline: {e}")
        return None


# ===================================================================
# HELPER — Carga de datos
# ===================================================================
@st.cache_data
def load_sample_corpus() -> Optional[pd.DataFrame]:
    """Carga el corpus de ejemplo corpus_demo_bilingue.csv."""
    if os.path.exists(SAMPLE_CSV):
        return pd.read_csv(SAMPLE_CSV)
    return None


def load_uploaded_csv(uploaded_file, sep: str = ",") -> Optional[pd.DataFrame]:
    """Carga un CSV o Excel subido por el usuario."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig', sep=sep, on_bad_lines='skip')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Formato no soportado. Usá CSV o Excel.")
            return None
    except Exception as e:
        st.error(f"❌ Error leyendo archivo: {e}")
        return None

    df.columns = df.columns.astype(str).str.strip()
    if 'text' not in df.columns:
        st.error("⚠️ El archivo debe tener una columna llamada 'text'.")
        return None

    df['text'] = df['text'].astype(str).apply(reparar_texto)
    return df


# ===================================================================
# HELPERS — Encoding y texto
# ===================================================================
def reparar_texto(texto: str) -> str:
    """Repara encoding latin1/cp1252 → utf-8 solo si el texto está corrupto."""
    try:
        texto.encode('latin1')
    except UnicodeEncodeError:
        return texto
    try:
        return texto.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            return texto.encode('cp1252').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return texto


def _es_cancelacion_suscripcion(texto: str) -> bool:
    """Detecta intención de cancelar suscripción o servicio."""
    tiene_cancel = any(kw in texto for kw in CANCELACION_KW)
    if not tiene_cancel:
        return False
    if any(kw in texto for kw in SUSCRIPCION_KW):
        return True
    if "cancel" in texto and any(kw in texto for kw in ["plan", "suscrip", "assinatur"]):
        return True
    # Detección general: "baja" + "no me interesa" sin menciones de pedido/cuenta
    if "baja" in texto and any(kw in texto for kw in ["no me interesa", "no quiero", "quiero darme"]):
        return True
    return False


def aplicar_reglas_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """Override de intención/sentimiento con reglas heurísticas post-pipeline."""
    nuevas_intents = []
    nuevos_sents = []

    for _, row in df.iterrows():
        texto_min = str(row.get("text", "")).lower()
        intent = row["intent"]
        sentiment = row["sentiment"]

        # 1. Cancelación de suscripción → cancelacion
        if _es_cancelacion_suscripcion(texto_min):
            intent = "cancelacion"

        # 2. Eliminación de cuenta → delete_account
        es_abandono = any(kw in texto_min for kw in DELETE_ACCOUNT_KW) and (
            "cuenta" in texto_min or "conta" in texto_min
        )
        if es_abandono:
            intent = "delete_account"

        # 3. Modificación de pedido → modificacion_pedido
        es_modificacion = False
        for and_group, or_group in MODIFY_ORDER_RULES:
            if any(a in texto_min for a in and_group) and any(o in texto_min for o in or_group):
                es_modificacion = True
                break
        if es_modificacion:
            intent = "modificacion_pedido"
            if sentiment == "negative":
                sentiment = "neutral"

        nuevas_intents.append(intent)
        nuevos_sents.append(sentiment)

    df["intent"] = nuevas_intents
    df["sentiment"] = nuevos_sents
    return df


# ===================================================================
# HELPERS — Análisis
# ===================================================================
def run_analysis(data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Ejecuta el pipeline completo sobre todos los textos del DataFrame.

    Returns:
        (results_df, model_status) — results_df con una fila por mensaje,
        columnas: text, sentiment, sentiment_prob, intent, intent_prob,
                  churn_score, sentiment_contrib, frustration_contrib, intent_contrib.
        model_status: dict con estado de modelos o None si falló.
    """
    pipeline = get_pipeline()
    if pipeline is None:
        return None, None

    # Validar columna 'text'
    if 'text' not in data.columns:
        st.error("❌ El CSV debe tener una columna 'text' con los mensajes.")
        return None, None

    texts = data['text'].fillna('').tolist()

    # batch_predict procesa todos los textos en orden
    results = pipeline.batch_predict(texts)

    # Build DataFrame
    rows = []
    for r in results:
        sent = r["sentiment"]
        intt = r["intent"]
        churn = r["churn"]
        rows.append({
            "text": r["text"],
            "sentiment": sent["label"],
            "sentiment_prob": sent["probability"],
            "sentiment_neg_prob": sent["probabilities"].get("negative", 0.0),
            "sentiment_neu_prob": sent["probabilities"].get("neutral", 0.0),
            "sentiment_pos_prob": sent["probabilities"].get("positive", 0.0),
            "intent": intt["intent"],
            "intent_prob": intt["probability"],
            "churn_score": churn["aggregate_score"],
            "sentiment_contrib": churn["sentiment_contribution"],
            "frustration_contrib": churn["frustration_contribution"],
            "intent_contrib": churn["intent_contribution"],
        })

    df_results = pd.DataFrame(rows)

    # Post-processing: reglas de negocio + macro-intenciones
    df_results = aplicar_reglas_negocio(df_results)
    df_results["macro_intent"] = df_results["intent"].map(
        lambda x: MACRO_INTENT_MAP.get(x, "Otras Consultas")
    )

    model_status = pipeline.get_model_status()

    return df_results, model_status


def has_fallback(status: dict) -> bool:
    """True si algún modelo está en modo fallback."""
    if status is None:
        return True
    return status.get("sentiment") == "fallback" or status.get("intent") == "fallback"


# ===================================================================
# VISUALIZACIONES
# ===================================================================
def show_metrics_row(df: pd.DataFrame):
    """Fila de métricas principales con 4 columnas."""
    st.header("📊 Métricas Principales")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Mensajes Procesados",
            f"{len(df):,}",
            help="Cantidad de mensajes analizados en la carga actual",
        )

    with col2:
        avg_churn = df["churn_score"].mean()
        st.metric(
            "Churn Score Promedio",
            f"{avg_churn:.3f}",
            delta=None,
            help="Score de riesgo de abandono promedio (0-1). Más alto = mayor riesgo.",
        )

    with col3:
        high_risk = (df["churn_score"] >= 0.5).sum()
        st.metric(
            "Alto Riesgo (≥0.5)",
            f"{high_risk:,}",
            delta=f"{high_risk / len(df) * 100:.1f}% del total",
            delta_color="inverse",
            help="Mensajes con churn score ≥ 0.5",
        )

    with col4:
        sent_dist = df["sentiment"].value_counts()
        neg_pct = sent_dist.get("negative", 0) / len(df) * 100
        st.metric(
            "Sentimiento Negativo",
            f"{neg_pct:.1f}%",
            delta=f"{sent_dist.get('negative', 0):,} mensajes",
            delta_color="inverse",
            help="Porcentaje de mensajes con sentimiento negativo",
        )


def show_sentiment_pie(df: pd.DataFrame):
    """Gráfico de torta — distribución de sentimiento 3 clases."""
    st.subheader("🎭 Distribución de Sentimiento")

    sent_counts = df["sentiment"].value_counts().reindex(
        ["negative", "neutral", "positive"], fill_value=0
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[s.capitalize() for s in sent_counts.index],
                values=sent_counts.values,
                marker=dict(
                    colors=[SENTIMENT_COLORS[s] for s in sent_counts.index]
                ),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{value:,} mensajes (%{percent})<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=350,
        margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_intent_bar(df: pd.DataFrame):
    """Gráfico de barras — distribución de intenciones 9 clases."""
    st.subheader("🎯 Distribución de Intenciones")

    intent_counts = df["intent"].value_counts().reindex(INTENT_LABELS, fill_value=0)
    intent_labels = [INTENT_DISPLAY.get(i, i.replace("_", " ").title()) for i in intent_counts.index]

    fig = go.Figure(
        data=[
            go.Bar(
                x=intent_labels,
                y=intent_counts.values,
                marker_color="#9b59b6",
                hovertemplate="<b>%{x}</b><br>%{y:,} mensajes<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=350,
        margin=dict(t=0, b=80, l=0, r=0),
        xaxis=dict(tickangle=45),
        yaxis=dict(title="Cantidad de Mensajes"),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_churn_histogram(df: pd.DataFrame) -> float:
    """Histograma de churn scores con slider de threshold.

    Returns:
        threshold actual (float).
    """
    st.header("⚠️ Distribución de Churn Score")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df["churn_score"],
                nbinsx=40,
                marker_color="#3498db",
                hovertemplate="Score: %{x:.2f}<br>Mensajes: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            height=350,
            xaxis=dict(title="Churn Score", range=[0, 1]),
            yaxis=dict(title="Cantidad de Mensajes"),
            margin=dict(t=0, b=0, l=0, r=0),
            bargap=0.05,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        threshold = st.slider(
            "Threshold Alto Riesgo",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Mensajes con churn score ≥ este valor se marcan como alto riesgo",
        )
        high_risk_count = (df["churn_score"] >= threshold).sum()
        st.metric(
            "Alto Riesgo",
            f"{high_risk_count:,}",
            delta=f"{high_risk_count / len(df) * 100:.1f}%",
            delta_color="inverse",
        )

    return threshold


def show_high_risk_table(df: pd.DataFrame, threshold: float):
    """Tabla de mensajes con churn score ≥ threshold, ordenable."""
    st.header(f"🔴 Mensajes de Alto Riesgo (Churn ≥ {threshold:.2f})")

    high_risk = df[df["churn_score"] >= threshold].copy()

    if high_risk.empty:
        st.success(f"✅ No hay mensajes con churn score ≥ {threshold:.2f}")
        return

    # Preparar columnas para mostrar
    display_cols = {
        "text": "Mensaje",
        "sentiment": "Sentimiento",
        "intent": "Intención",
        "macro_intent": "Macro-Int.",
        "churn_score": "Churn Score",
        "sentiment_contrib": "Contrib. Sent.",
        "frustration_contrib": "Contrib. Frust.",
        "intent_contrib": "Contrib. Intent",
    }

    display_df = high_risk[list(display_cols.keys())].copy()
    display_df.rename(columns=display_cols, inplace=True)

    # Formatear columnas numéricas
    for col in ["Churn Score", "Contrib. Sent.", "Contrib. Frust.", "Contrib. Intent"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")

    # Truncar texto para previsualización
    display_df["Mensaje"] = display_df["Mensaje"].apply(
        lambda x: x[:100] + "..." if isinstance(x, str) and len(x) > 100 else x
    )

    # Badge de sentimiento con color
    def sentiment_badge(sent: str) -> str:
        emoji_map = {"negative": "🔴", "neutral": "🟡", "positive": "🟢"}
        return f"{emoji_map.get(sent, '⚪')} {sent.capitalize()}"

    display_df["Sentimiento"] = display_df["Sentimiento"].apply(sentiment_badge)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "Mensaje": st.column_config.TextColumn("Mensaje", width="large"),
            "Churn Score": st.column_config.TextColumn("Churn Score", width="small"),
            "Sentimiento": st.column_config.TextColumn("Sentimiento", width="small"),
        },
    )

    # Botón de descarga
    csv = high_risk.to_csv(index=False)
    st.download_button(
        "📥 Descargar CSV de Alto Riesgo",
        csv,
        f"conversaai_high_risk_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )


def show_stacked_intent_chart(df: pd.DataFrame):
    """Stacked bar de intenciones filtrable por sentimiento."""
    st.header("📊 Intenciones por Sentimiento")

    # Filtro de sentimiento
    available_sentiments = df["sentiment"].unique().tolist()
    default_sentiments = available_sentiments.copy()

    selected_sentiments = st.multiselect(
        "Filtrar por Sentimiento",
        options=["negative", "neutral", "positive"],
        default=default_sentiments,
        help="Seleccioná uno o más sentimientos para filtrar las intenciones",
    )

    # Filtrar
    if selected_sentiments:
        filtered = df[df["sentiment"].isin(selected_sentiments)]
    else:
        filtered = df

    # Crosstab: intents × sentiment
    crosstab = pd.crosstab(
        filtered["intent"], filtered["sentiment"],
        normalize="index",
    )

    # Reindex para asegurar que todas las intenciones aparezcan
    for s in ["negative", "neutral", "positive"]:
        if s not in crosstab.columns:
            crosstab[s] = 0.0

    crosstab = crosstab.reindex(INTENT_LABELS, fill_value=0.0)
    intent_labels = [INTENT_DISPLAY.get(i, i.replace("_", " ").title()) for i in crosstab.index]

    # Stacked bar horizontal
    fig = go.Figure()

    for sent in ["negative", "neutral", "positive"]:
        fig.add_trace(
            go.Bar(
                name=sent.capitalize(),
                y=intent_labels,
                x=crosstab[sent].values,
                orientation="h",
                marker_color=SENTIMENT_COLORS[sent],
                hovertemplate="<b>%{y}</b><br>"
                              + f"{sent.capitalize()}" + ": %{x:.1%}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        height=400,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(title="Proporción", tickformat=".0%"),
        yaxis=dict(title="Intención"),
        legend=dict(title="Sentimiento", orientation="h", yanchor="bottom", y=-0.15),
        hovermode="y unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def show_temporal_evolution(df: pd.DataFrame):
    """Gráfico de línea — evolución de frustración por idioma (últimos 30 días simulados)."""
    st.header("📈 Evolución Temporal de la Frustración")

    df_plot = df.copy()
    if 'date' not in df_plot.columns:
        hoy = datetime.now()
        df_plot['date'] = [hoy - timedelta(days=random.randint(0, 29)) for _ in range(len(df_plot))]
    if 'lang' not in df_plot.columns:
        df_plot['lang'] = random.choices(['es', 'pt'], k=len(df_plot))

    df_plot['Idioma'] = df_plot['lang'].astype(str).str.lower().map(
        {'es': 'Español', 'pt': 'Portugués', 'pt-br': 'Portugués'}
    ).fillna('Otro')

    df_daily = df_plot.groupby(['date', 'Idioma']).size().reset_index(name='total')
    df_neg = df_plot[df_plot['sentiment'] == 'negative'].groupby(
        ['date', 'Idioma']
    ).size().reset_index(name='neg')

    trend = pd.merge(df_daily, df_neg, on=['date', 'Idioma'], how='left').fillna(0)
    trend['Frustración (%)'] = (trend['neg'] / trend['total']) * 100
    trend = trend.sort_values('date')

    fig = px.line(
        trend, x='date', y='Frustración (%)', color='Idioma',
        title="Evolución de Frustración Promedio (Últimos 30 días)",
        markers=True,
        color_discrete_map={'Español': '#3498db', 'Portugués': '#2ecc71'},
    )
    fig.update_layout(
        height=350,
        margin=dict(t=30, b=0, l=0, r=0),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)


def show_macro_intent_chart(df: pd.DataFrame):
    """Gráfico de barras — distribución de macro-intenciones."""
    st.subheader("📊 Volumen por Macro-Intención")
    macro_counts = df['macro_intent'].value_counts().reset_index()
    macro_counts.columns = ['Macro-Intención', 'Cantidad']

    fig = px.bar(
        macro_counts, x='Cantidad', y='Macro-Intención', orientation='h',
        text_auto=True, color_discrete_sequence=['#3498db'],
    )
    fig.update_layout(
        height=300,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Volumen", yaxis_title="",
        margin=dict(t=0, b=0, l=0, r=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_explorer(df: pd.DataFrame):
    """Explorador de mensajes con filtros (legacy, actualizado)."""
    st.header("🔍 Explorador de Mensajes")

    col1, col2, col3 = st.columns(3)

    with col1:
        filtro_sentimiento = st.selectbox(
            "Sentimiento",
            ["Todos"] + ["negative", "neutral", "positive"],
            key="explorer_sentiment",
        )

    with col2:
        filtro_intent = st.selectbox(
            "Intención",
            ["Todos"] + INTENT_LABELS,
            format_func=lambda x: INTENT_DISPLAY.get(x, x.replace("_", " ").title()) if x != "Todos" else x,
            key="explorer_intent",
        )

    with col3:
        min_churn, max_churn = st.slider(
            "Rango Churn Score",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05,
            key="explorer_churn",
        )

    # Aplicar filtros
    df_filtrado = df.copy()

    if filtro_sentimiento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["sentiment"] == filtro_sentimiento]

    if filtro_intent != "Todos":
        df_filtrado = df_filtrado[df_filtrado["intent"] == filtro_intent]

    df_filtrado = df_filtrado[
        (df_filtrado["churn_score"] >= min_churn)
        & (df_filtrado["churn_score"] <= max_churn)
    ]

    st.write(f"### {len(df_filtrado):,} mensajes encontrados")

    # Mostrar tabla
    display_explorer = df_filtrado[["text", "sentiment", "intent", "macro_intent", "churn_score"]].copy()
    display_explorer.columns = ["Mensaje", "Sentimiento", "Intención", "Macro-Intención", "Churn Score"]

    display_explorer["Sentimiento"] = display_explorer["Sentimiento"].apply(
        lambda s: f"{'🔴' if s=='negative' else '🟡' if s=='neutral' else '🟢'} {s.capitalize()}"
    )
    display_explorer["Intención"] = display_explorer["Intención"].apply(
        lambda i: INTENT_DISPLAY.get(i, i.replace("_", " ").title())
    )
    display_explorer["Churn Score"] = display_explorer["Churn Score"].apply(
        lambda x: f"{x:.3f}"
    )

    st.dataframe(
        display_explorer.head(100),
        use_container_width=True,
        height=400,
    )

    csv = df_filtrado.to_csv(index=False)
    st.download_button(
        "📥 Descargar CSV filtrado",
        csv,
        f"conversaai_filtrado_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )


# ===================================================================
# SIDEBAR
# ===================================================================
def sidebar() -> str:
    """Configura el sidebar. Retorna la página seleccionada."""
    st.sidebar.title("💬 ConversaAI")
    st.sidebar.markdown("### Pipeline de Análisis")

    # ── Data source ──
    data_source = st.sidebar.radio(
        "Fuente de datos",
        ["📂 Corpus de ejemplo", "📁 Subir CSV"],
        index=0,
        help="Seleccioná 'Corpus de ejemplo' para usar el dataset precargado "
             "o 'Subir CSV' para analizar tus propios datos.",
    )

    data = None
    if data_source == "📂 Corpus de ejemplo":
        data = load_sample_corpus()
        if data is not None:
            st.sidebar.success(f"✅ Corpus cargado: {len(data):,} mensajes")
            # Auto-analysis al cargar corpus demo (solo la primera vez)
            if not st.session_state["corpus_auto_analyzed"]:
                st.session_state["corpus_auto_analyzed"] = True
                with st.sidebar.status("Analizando corpus demo...", expanded=True) as status:
                    df_results, model_status = run_analysis(data)
                    if df_results is not None and not df_results.empty:
                        st.session_state["results_df"] = df_results
                        st.session_state["model_status"] = model_status
                        st.session_state["last_update"] = datetime.now().strftime('%Y-%m-%d %H:%M')
                        status.update(label="✅ Análisis completo", state="complete")
                    elif df_results is not None and df_results.empty:
                        st.sidebar.warning("El análisis devolvió 0 resultados")
                        status.update(label="Sin resultados", state="error")
                    else:
                        status.update(label="❌ Error en análisis", state="error")
        else:
            st.sidebar.warning("⚠️ No se encontró corpus_demo_bilingue.csv")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Subí tu archivo",
            type=["csv", "xlsx", "xls"],
            help="El archivo debe tener una columna 'text' con los mensajes",
        )
        if uploaded_file is not None:
            sep = ","
            if uploaded_file.name.endswith('.csv'):
                sep = st.sidebar.radio("Separador del CSV:", [",", ";"], horizontal=True)
            data = load_uploaded_csv(uploaded_file, sep=sep)
            if data is not None:
                st.sidebar.success(f"✅ Archivo cargado: {len(data):,} mensajes")

    # ── Run Analysis (manual, solo para CSV) ──
    if data_source == "📁 Subir CSV":
        st.sidebar.divider()
        run_button = st.sidebar.button(
            "▶️ Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=data is None,
            help="Procesar todos los mensajes con el pipeline de IA",
        )

        if run_button and data is not None:
            with st.sidebar.status("Analizando...", expanded=True) as status:
                df_results, model_status = run_analysis(data)
                if df_results is not None and not df_results.empty:
                    st.session_state["results_df"] = df_results
                    st.session_state["model_status"] = model_status
                    st.session_state["last_update"] = datetime.now().strftime('%Y-%m-%d %H:%M')
                    status.update(label="✅ Análisis completo", state="complete")
                elif df_results is not None and df_results.empty:
                    st.sidebar.warning("El análisis devolvió 0 resultados")
                    status.update(label="Sin resultados", state="error")
                else:
                    status.update(label="❌ Error en análisis", state="error")

    # ── Model status ──
    st.sidebar.divider()
    st.sidebar.markdown("### Estado del Pipeline")

    if "model_status" in st.session_state and st.session_state["model_status"]:
        status = st.session_state["model_status"]
        for model, mode in status.items():
            emoji = "🤖" if mode == "xlm-r" else "⚙️"
            label = "XLM-R" if mode == "xlm-r" else "Fallback"
            st.sidebar.markdown(f"{emoji} **{model.capitalize()}**: {label}")
    else:
        st.sidebar.info("ℹ️ Ejecutá 'Run Analysis' para ver el estado")

    # ── Navigation ──
    st.sidebar.divider()
    st.sidebar.markdown("### Navegación")
    page = st.sidebar.radio(
        "Ir a:",
        ["📊 Dashboard", "🔍 Explorar", "💡 Acciones"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    last_update = st.session_state.get("last_update", None)
    if last_update:
        st.sidebar.markdown(f"**Actualizado:** {last_update}")
    else:
        st.sidebar.markdown("**Actualizado:** —")
    st.sidebar.caption(
        "Desarrollado por ConversaAI · "
        "Modelos: [Rosela/xlm-r-sentiment-espt](https://huggingface.co/Rosela/xlm-r-sentiment-espt)"
    )

    return page


# ===================================================================
# MAIN
# ===================================================================
def main():
    # ── Initialize session state ──
    if "results_df" not in st.session_state:
        st.session_state["results_df"] = None
    if "model_status" not in st.session_state:
        st.session_state["model_status"] = None
    if "last_update" not in st.session_state:
        st.session_state["last_update"] = None
    if "corpus_auto_analyzed" not in st.session_state:
        st.session_state["corpus_auto_analyzed"] = False

    # ── Sidebar ──
    page = sidebar()

    # ── Pipeline check ──
    if not _HAS_PIPELINE:
        with st.container():
            st.title("💬 ConversaAI — Dashboard de Insights")
            st.error(
                f"❌ No se pudo cargar el pipeline ConversaAIPipeline. "
                f"Error: {_PIPELINE_ERR}. "
                "Asegurate de tener instaladas las dependencias: "
                "`pip install -r requirements.txt`"
            )
        return

    # ── No results yet? Show welcome/onboarding screen ──
    if st.session_state["results_df"] is None:
        st.title("💬 ConversaAI — Dashboard de Insights")
        st.markdown(
            "### Sistema de análisis de conversaciones de soporte\n\n"
            "Cargá datos desde el panel izquierdo y presioná **▶️ Run Analysis** "
            "para comenzar.\n\n"
            "El pipeline clasifica cada mensaje en:\n"
            "- **Sentimiento** (3 clases: negativo, neutral, positivo) vía XLM-R fine-tuneado (92.2% acc)\n"
            "- **Intención** (9 clases: cancelación, queja, reembolso, etc.) vía XLM-R fine-tuneado (99.6% F1)\n"
            "- **Churn Score** (0–1) combinando sentimiento, frustración e intención\n\n"
            "📂 Usá el *Corpus de ejemplo* precargado o subí tu propio CSV."
        )
        return

    # ── We have results! ──
    df = st.session_state["results_df"]
    status = st.session_state["model_status"]

    # ── Model status banner ──
    if status and not has_fallback(status):
        st.success("✅ **Modo completo**: XLM-R fine-tuneado activo para sentimiento e intención")
    elif status:
        st.warning(
            "⚠️ **Modelos en modo fallback.** "
            "Los modelos fine-tuneados no están disponibles localmente."
        )

    if page.startswith("📊 Dashboard"):
        # ── Header ──
        st.title("💬 ConversaAI — Dashboard de Insights")
        st.markdown("### Análisis de conversaciones de soporte")

        if status and not has_fallback(status):
            st.success("✅ **XLM-R fine-tuneado** activo para sentimiento (92.2%) e intención (99.6% F1)")
        elif status:
            st.info("⚙️ **Modo fallback** — modelos fine-tuneados no disponibles")

        # ── Métricas principales ──
        show_metrics_row(df)

        st.divider()

        # ── Gráficos: sentimiento + intención ──
        col_left, col_right = st.columns(2)

        with col_left:
            show_sentiment_pie(df)

        with col_right:
            show_intent_bar(df)

        st.divider()

        # ── Churn histograma + tabla ──
        threshold = show_churn_histogram(df)

        st.divider()

        show_high_risk_table(df, threshold)

        st.divider()

        # ── Macro-intenciones ──
        show_macro_intent_chart(df)

        st.divider()

        # ── Stacked bar ──
        show_stacked_intent_chart(df)

        st.divider()

        # ── Evolución temporal ──
        show_temporal_evolution(df)

    elif page.startswith("🔍 Explorar"):
        st.title("🔍 Explorador de Mensajes")
        show_explorer(df)

    elif page.startswith("💡 Acciones"):
        st.title("💡 Recomendaciones para el Equipo de Producto")

        if df is not None:
            # ── Churn por intent ──
            st.markdown("### 📊 Churn Promedio por Intención")
            intent_churn = df.groupby("intent")["churn_score"].mean().reindex(INTENT_LABELS, fill_value=0)
            intent_labels = [INTENT_DISPLAY.get(i, i.replace("_", " ").title()) for i in intent_churn.index]
            churn_overall = df["churn_score"].mean()

            fig = go.Figure()
            colors = ["#e74c3c" if v > churn_overall else "#2ecc71" for v in intent_churn.values]
            fig.add_trace(go.Bar(x=intent_labels, y=intent_churn.values, marker_color=colors))
            fig.add_hline(y=churn_overall, line_dash="dash", line_color="#7f8c8d",
                          annotation_text=f"Promedio general: {churn_overall:.3f}")
            fig.update_layout(height=350, xaxis=dict(tickangle=45), margin=dict(t=0, b=80, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

            st.caption("🔴 Barras rojas = por encima del promedio. Priorizar estas intenciones.")

            st.divider()

            # ── Riesgo por intent ──
            st.markdown("### 🎯 Intenciones de Alto Riesgo")
            col1, col2 = st.columns(2)

            with col1:
                top_negative = (
                    df[df["sentiment"] == "negative"]["intent"]
                    .value_counts()
                    .head(5)
                )
                st.markdown("**Más sentimiento negativo:**")
                for intent, count in top_negative.items():
                    display = INTENT_DISPLAY.get(intent, intent.replace("_", " ").title())
                    st.write(f"- {display}: {count:,} mensajes")

            with col2:
                high_churn_intents = (
                    df[df["churn_score"] >= 0.5]["intent"]
                    .value_counts()
                    .head(5)
                )
                st.markdown("**Mayor churn (≥0.5):**")
                for intent, count in high_churn_intents.items():
                    display = INTENT_DISPLAY.get(intent, intent.replace("_", " ").title())
                    st.write(f"- {display}: {count:,} mensajes")

            st.divider()

            # ── Contribución al churn ──
            st.markdown("### 🔍 Desglose de Churn por Intención")
            contrib = df.groupby("intent")[["sentiment_contrib", "frustration_contrib", "intent_contrib"]].mean()
            contrib = contrib.reindex(INTENT_LABELS, fill_value=0)
            intent_labels_contrib = [INTENT_DISPLAY.get(i, i.replace("_", " ").title()) for i in contrib.index]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Sentimiento", x=intent_labels_contrib, y=contrib["sentiment_contrib"],
                                 marker_color="#3498db"))
            fig.add_trace(go.Bar(name="Frustración", x=intent_labels_contrib, y=contrib["frustration_contrib"],
                                 marker_color="#e74c3c"))
            fig.add_trace(go.Bar(name="Intención", x=intent_labels_contrib, y=contrib["intent_contrib"],
                                 marker_color="#9b59b6"))
            fig.update_layout(barmode="stack", height=350, xaxis=dict(tickangle=45), margin=dict(t=0, b=80, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Azul = sentimiento negativo · Rojo = frustración por keywords · Violeta = intención de riesgo")

            st.divider()

            # ── Alertas (basadas en intención, como app.py) ──
            st.markdown("### 🚨 Alertas y Acciones Sugeridas")
            alerts = []
            MICRO_INTENT = df['intent'].astype(str).str.lower()

            for label, key, emoji, msg_yes, msg_no in [
                ("Cancelación", {"cancelacion"}, "🔴",
                 "Revisar proceso de retención — ofrecer descuentos o beneficios", "0 casos"),
                ("Queja", {"queja"}, "🔴",
                 "Escalar a equipo de calidad — contactar al cliente en < 24h", "0 casos"),
                ("Reembolso", {"reembolso", "facturacion_pago"}, "🟠",
                 "Agilizar proceso de reembolso — priorizar sobre otros tickets", "0 casos"),
                ("Alta frustración", {"queja", "cancelacion", "reembolso"}, "🔴",
                 "Evaluar calidad del servicio — revisar conversaciones recientes", "0 casos"),
            ]:
                subset = df[MICRO_INTENT.isin(key)]
                casos = len(subset)
                churn_prom = subset['churn_score'].mean() if casos > 0 else 0.0
                if casos > 0:
                    st.markdown(f"**{emoji} {label}** — {casos} casos (churn prom. {churn_prom:.2f})")
                    st.markdown(f"{emoji} **{msg_yes}**")
                else:
                    st.markdown(f"**✅ {label}** — {msg_no}")
                st.markdown("---")

            st.divider()

            # ── Resumen ──
            st.markdown("### 📋 Resumen Ejecutivo")
            col1, col2, col3 = st.columns(3)
            with col1:
                total = len(df)
                pct_neg = (df["sentiment"] == "negative").mean() * 100
                st.metric("Total mensajes", f"{total:,}", help="Mensajes analizados en esta carga")
            with col2:
                st.metric("Sentimiento negativo", f"{pct_neg:.1f}%",
                          help="Porcentaje de mensajes con sentimiento negativo")
            with col3:
                high_risk_pct = (df["churn_score"] >= 0.5).mean() * 100
                st.metric("Alto riesgo de churn", f"{high_risk_pct:.1f}%",
                          help="Porcentaje de mensajes con churn ≥ 0.5")

        else:
            st.info("Ejecutá **Run Analysis** desde el panel izquierdo para ver recomendaciones.")


# ===================================================================
# ENTRY POINT
# ===================================================================
if __name__ == "__main__":
    main()
