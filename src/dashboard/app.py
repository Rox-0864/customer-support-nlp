import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.xlmr_sentiment import classify as classify_sentiment
from src.models.intent_classifier import classify_intent

st.set_page_config(page_title="ConversaAI - Support Analytics", layout="wide")

st.title("ConversaAI - Análisis de Conversaciones de Soporte")
st.markdown("Dashboard con modelo **XLM-R fine-tuned** en 271k reviews ES/PT.")

# Sidebar
with st.sidebar:
    st.header("Filtros")

    lang_filter = st.selectbox(
        "Idioma",
        options=["Todos", "Español", "Portugués"],
        index=0
    )

    date_range = st.date_input(
        "Rango de Fechas",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )

    risk_filter = st.multiselect(
        "Riesgo de Churn",
        options=["low", "medium", "high", "critical"],
        default=["low", "medium", "high", "critical"]
    )

    st.markdown("---")
    st.markdown("### Probar modelo en vivo")
    test_text = st.text_area("Escribí un mensaje de cliente:", height=100)
    test_lang = st.selectbox("Idioma (opcional)", ["auto", "es", "pt"], index=0)
    if st.button("Analizar") and test_text:
        with st.spinner("Analizando..."):
            lang_param = None if test_lang == "auto" else test_lang
            sr = classify_sentiment(test_text, lang=lang_param)
            ir = classify_intent(test_text, lang=lang_param)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentimiento", sr.label.upper(), f"{sr.confidence:.1%}")
            with col2:
                st.metric("Intención", ir.intent, f"{ir.confidence:.1%}")
            st.caption(f"Idioma detectado: {sr.lang} | Modelo: {ir.model}")

# Load data
@st.cache_data
def load_conversation_data():
    data_path = "data/support_conversations.csv"
    if not os.path.exists(data_path):
        st.error("No se encontró data/support_conversations.csv.")
        return None

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_conversation_data()

if df is None:
    st.stop()

# Apply filters
if lang_filter == "Español":
    df = df[df['lang'] == 'es']
elif lang_filter == "Portugués":
    df = df[df['lang'] == 'pt']

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Resumen General",
    "Intenciones No Resueltas",
    "Análisis de Frustración",
    "Recomendaciones"
])

# Tab 1: General Summary
with tab1:
    st.header("Resumen General de Conversaciones")

    col1, col2, col3, col4 = st.columns(4)

    total_messages = len(df)
    unique_threads = df['thread_id'].nunique()

    frustration_map = {"none": 0.0, "low": 0.2, "medium": 0.5, "high": 0.8}
    df['frustration_numeric'] = df['frustration_level'].map(frustration_map)
    avg_frustration = df['frustration_numeric'].mean()

    high_risk_threads = df[df['frustration_level'] == 'high']['thread_id'].nunique()

    with col1:
        st.metric("Total Mensajes", total_messages)
    with col2:
        st.metric("Conversaciones", unique_threads)
    with col3:
        st.metric("Frustración Promedio", f"{avg_frustration:.2f}")
    with col4:
        st.metric("Conversaciones de Alto Riesgo", high_risk_threads)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de Intenciones")
        intent_counts = df[df['sender'] == 'customer']['intent'].value_counts()
        fig_intent = px.pie(
            values=intent_counts.values,
            names=intent_counts.index,
            title="Intenciones de Clientes"
        )
        st.plotly_chart(fig_intent)

    with col2:
        st.subheader("Distribución de Sentimiento")
        sentiment_counts = df[df['sender'] == 'customer']['sentiment'].value_counts()
        fig_sentiment = px.bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Sentimiento de Clientes",
            color=sentiment_counts.index,
            color_discrete_map={"positive": "green", "negative": "red", "neutral": "gray"}
        )
        st.plotly_chart(fig_sentiment)

    st.subheader("Mensajes por Día")
    df['date'] = df['timestamp'].dt.date
    messages_per_day = df.groupby(['date', 'lang']).size().reset_index(name='count')
    fig_time = px.line(
        messages_per_day,
        x='date',
        y='count',
        color='lang',
        title="Volumen de Mensajes por Día"
    )
    st.plotly_chart(fig_time)

# Tab 2: Unresolved Intents
with tab2:
    st.header("Intenciones No Resueltas")
    st.markdown("Intenciones que fueron mencionadas pero no tienen un mensaje de resolución.")

    threads = df.groupby('thread_id')

    unresolved_data = []
    for thread_id, thread_df in threads:
        customer_msgs = thread_df[thread_df['sender'] == 'customer']
        intents_mentioned = customer_msgs['intent'].unique()
        resolution_intents = ['resuelto', 'resolvido']
        has_resolution = thread_df[thread_df['intent'].isin(resolution_intents)].shape[0] > 0

        if not has_resolution:
            lang = thread_df['lang'].iloc[0]
            frustration = thread_df['frustration_level'].mode()[0] if len(thread_df) > 0 else 'unknown'

            for intent in intents_mentioned:
                unresolved_data.append({
                    'thread_id': thread_id,
                    'lang': lang,
                    'intent': intent,
                    'frustration_level': frustration,
                    'message_count': len(thread_df)
                })

    if unresolved_data:
        unresolved_df = pd.DataFrame(unresolved_data)
        risk_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
        unresolved_df['risk_score'] = unresolved_df['frustration_level'].map(risk_map)
        unresolved_df = unresolved_df.sort_values('risk_score', ascending=False)

        st.subheader(f"Total: {len(unresolved_df)} intenciones no resueltas")

        intent_summary = unresolved_df.groupby('intent').agg({
            'thread_id': 'count',
            'lang': lambda x: list(x.unique())
        }).rename(columns={'thread_id': 'count'})

        st.dataframe(intent_summary)

        st.subheader("Detalle por Conversación")
        display_df = unresolved_df[['thread_id', 'lang', 'intent', 'frustration_level', 'message_count']]
        st.dataframe(display_df)
    else:
        st.success("Todas las intenciones han sido resueltas!")

# Tab 3: Frustration Analysis
with tab3:
    st.header("Análisis de Frustración")

    st.subheader("Frustración por Intención")
    frustration_by_intent = df.groupby(['intent', 'frustration_level']).size().reset_index(name='count')
    fig = px.bar(
        frustration_by_intent,
        x='intent',
        y='count',
        color='frustration_level',
        title="Niveles de Frustración por Intención",
        color_discrete_map={"none": "green", "low": "yellow", "medium": "orange", "high": "red"}
    )
    st.plotly_chart(fig)

    st.subheader("Evolución de Frustración en el Tiempo")
    df['hour'] = df['timestamp'].dt.floor('h')
    frustration_timeline = df.groupby(['hour', 'lang'])['frustration_numeric'].mean().reset_index()
    fig_time = px.line(
        frustration_timeline,
        x='hour',
        y='frustration_numeric',
        color='lang',
        title="Frustración Promedio por Hora",
        labels={'frustration_numeric': 'Frustración Promedio'}
    )
    st.plotly_chart(fig_time)

    st.subheader("Conversaciones con Mayor Frustración")
    thread_frustration = df.groupby('thread_id').agg({
        'frustration_numeric': 'mean',
        'lang': 'first',
        'intent': lambda x: list(x.unique())
    }).reset_index()
    thread_frustration = thread_frustration.sort_values('frustration_numeric', ascending=False).head(10)
    thread_frustration.columns = ['Thread ID', 'Frustración Promedio', 'Idioma', 'Intenciones']
    st.dataframe(thread_frustration)

# Tab 4: Recommendations
with tab4:
    st.header("Recomendaciones Prácticas")
    st.markdown("Basado en el análisis de conversaciones no resueltas y niveles de frustración.")

    recommendations = []

    cancel_intents = df[df['intent'].isin(['cancelacion', 'cancelamento']) & (df['sender'] == 'customer')]
    if len(cancel_intents) > 0:
        cancel_threads = cancel_intents['thread_id'].nunique()
        recommendations.append({
            'prioridad': 'CRÍTICA',
            'recomendación': f'Ofrecer retención proactiva: {cancel_threads} conversaciones con intención de cancelación detectadas',
            'acción': 'Contactar clientes con ofertas de descuento o mejora de servicio'
        })

    high_frustration_threads = df[df['frustration_level'] == 'high']['thread_id'].nunique()
    if high_frustration_threads > 0:
        recommendations.append({
            'prioridad': 'ALTA',
            'recomendación': f'Revisar calidad de atención: {high_frustration_threads} conversaciones con frustración alta',
            'acción': 'Asignar agentes senior y revisar scripts de respuesta'
        })

    tech_problems = df[(df['intent'] == 'problema_tecnico') & (df['sender'] == 'customer')]['thread_id'].nunique()
    if tech_problems > 0:
        recommendations.append({
            'prioridad': 'MEDIA',
            'recomendación': f'Problemas técnicos recurrentes: {tech_problems} conversaciones',
            'acción': 'Escalar a equipo técnico y crear base de conocimientos'
        })

    billing_intents = df[df['intent'].isin(['facturacion', 'faturamento']) & (df['sender'] == 'customer')]
    if len(billing_intents) > 0:
        recommendations.append({
            'prioridad': 'MEDIA',
            'recomendación': f'Consultas de facturación: {len(billing_intents)} mensajes',
            'acción': 'Automatizar respuestas de facturación o mejorar claridad de facturas'
        })

    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, hide_index=True)

        csv = rec_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Informe en CSV",
            data=csv,
            file_name=f"recomendaciones_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    else:
        st.info("No hay recomendaciones pendientes en el período seleccionado.")

st.markdown("---")
st.markdown("**ConversaAI** - XLM-R fine-tuned | Sentimiento: 271k reviews ES/PT")
