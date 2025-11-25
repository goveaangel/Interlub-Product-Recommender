import streamlit as st
import pandas as pd

from scripts.recomendador import (
    df_interlub_raw,
    recomendar_interlub_pro,
    plot_ranking,
    plot_radar_profile,
)

# ---------------------------
# Configuración básica de la app
# ---------------------------
st.set_page_config(
    page_title="Recomendador de Grasas Interlub",
    layout="wide"
)

st.title("🧪 Recomendador de Grasas Interlub")
st.markdown(
    """
    Esta app recibe requisitos básicos de operación (temperatura, carga, ambiente con agua)
    y devuelve las **grasas Interlub más recomendadas**, junto con un pequeño análisis.
    """
)

# ---------------------------
# Sidebar: parámetros del cliente
# ---------------------------
st.sidebar.header("Parámetros del cliente")

# Rango de temperatura
T_min = st.sidebar.slider(
    "Temperatura mínima requerida (°C)",
    min_value=-60,
    max_value=50,
    value=-10,
    step=5,
)

T_max = st.sidebar.slider(
    "Temperatura máxima requerida (°C)",
    min_value=60,
    max_value=250,
    value=130,
    step=10,
)

# Ambiente con agua
ambiente_agua = st.sidebar.checkbox(
    "Ambiente con presencia de agua (lavado / humedad)",
    value=True,
)

# Carga mecánica
op_carga = st.sidebar.selectbox(
    "Nivel de carga mecánica",
    options=["No especificar", "media", "alta", "extrema"],
    index=2,  # por defecto "alta"
)

carga = None if op_carga == "No especificar" else op_carga

# Cuántas recomendaciones mostrar
top_n = st.sidebar.slider(
    "Número de recomendaciones a mostrar",
    min_value=3,
    max_value=10,
    value=5,
)

# Botón para ejecutar
st.sidebar.markdown("---")
ejecutar = st.sidebar.button("🔍 Ejecutar recomendador")

# ---------------------------
# Lógica principal
# ---------------------------

if ejecutar:
    # 1. Construir diccionario de requerimientos
    req = {
        "T_min": T_min,
        "T_max": T_max,
        "ambiente_agua": ambiente_agua,
        "carga": carga,
    }

    with st.spinner("Calculando recomendaciones..."):
        df_top, fila_cliente_raw = recomendar_interlub_pro(
            req=req,
            top_n=top_n
        )

    if df_top is None:
        st.warning("No se encontraron grasas que cumplan los requisitos actuales.")
    else:
        st.success("✅ Recomendaciones generadas correctamente.")

        # ---------------------------
        # 2. Tabla resumen
        # ---------------------------
        st.subheader("📋 Top grasas recomendadas")

        cols_resumen = [
            "Temperatura de Servicio °C, min",
            "Temperatura de Servicio °C, max",
            "Punto de Soldadura Cuatro Bolas, kgf",
            "Resistencia al Lavado por Agua a 80°C, %",
            "Viscosidad del Aceite Base a 40°C. cSt",
            "sim_global",
            "sim_temp",
            "sim_carga",
            "sim_agua",
            "sim_visc",
            "score_norm",
            "explicacion_score",
        ]

        cols_resumen = [c for c in cols_resumen if c in df_top.columns]

        st.dataframe(
            df_top[cols_resumen].style.format(
                {
                    "sim_global": "{:.3f}",
                    "sim_temp": "{:.3f}",
                    "sim_carga": "{:.3f}",
                    "sim_agua": "{:.3f}",
                    "sim_visc": "{:.3f}",
                    "score_norm": "{:.1f}",
                }
            ),
            use_container_width=True,
        )

        # ---------------------------
        # 3. Gráfica de ranking (barras)
        # ---------------------------
        st.subheader("🏆 Ranking (score normalizado)")

        fig_bar = plot_ranking(df_top, score_col="score_norm")
        st.pyplot(fig_bar)

        # ---------------------------
        # 4. Radar chart para una grasa específica
        # ---------------------------
        st.subheader("📡 Comparación detalle vs perfil ideal")

        # Elegir cuál grasa graficar
        idx_opcion = st.selectbox(
            "Selecciona la grasa para ver su radar:",
            options=list(df_top.index),
            index=0,
            format_func=lambda x: f"Grasa #{x}",
        )

        fila_producto = df_top.loc[idx_opcion]

        numeric_cols = [
            "Temperatura de Servicio °C, min",
            "Temperatura de Servicio °C, max",
            "Resistencia al Lavado por Agua a 80°C, %",
            "Punto de Soldadura Cuatro Bolas, kgf",
            "Carga Timken Ok, lb",
        ]
        numeric_cols = [c for c in numeric_cols if c in df_interlub_raw.columns]

        fila_ideal = fila_cliente_raw.iloc[0]

        fig_radar = plot_radar_profile(
            df_interlub_raw,
            fila_ideal,
            fila_producto,
            numeric_cols,
            title=f"Radar de Grasa #{idx_opcion}",
        )
        st.pyplot(fig_radar)

else:
    st.info("Configura los parámetros en la barra lateral y presiona **'Ejecutar recomendador'**.")