import streamlit as st

from scripts.recomendador import (
    df_interlub_raw,
    recomendar_interlub_pro,
    plot_ranking,
    plot_radar_profile,
)

st.title("2️⃣ Recomendador de grasas Interlub")

st.markdown("""
Usando los parámetros definidos en **1_Parametros_del_cliente**, 
el sistema calcula el **ranking de grasas Interlub** que mejor se ajustan
al perfil del cliente.
""")

# ---------------------------
# Verificar parámetros
# ---------------------------
if "req" not in st.session_state or "top_n" not in st.session_state:
    st.warning(
        "Primero ve a **Parametros del cliente** para definir y guardar los parámetros."
    )
    st.stop()

req = st.session_state["req"]
top_n = st.session_state["top_n"]

st.subheader("📌 Parámetros en uso")

T_min = req["T_min"]
T_max = req["T_max"]
agua = "Sí" if req["ambiente_agua"] else "No"
carga = req["carga"] if req["carga"] is not None else "No especificada"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Temperatura mínima", f"{T_min} °C")

with col2:
    st.metric("Temperatura máxima", f"{T_max} °C")

with col3:
    st.metric("Presencia de agua", agua)

with col4:
    st.metric("Carga mecánica", carga)

st.markdown("---")

# ---------------------------
# Botón para ejecutar recomendador
# ---------------------------
ejecutar = st.button("🔍 Ejecutar recomendador")

if not ejecutar:
    st.info("Presiona el botón para generar las recomendaciones con estos parámetros.")
    st.stop()

# ---------------------------
# Llamar al backend
# ---------------------------
with st.spinner("Calculando recomendaciones..."):
    df_top, fila_cliente_raw = recomendar_interlub_pro(
        req=req,
        top_n=top_n
    )

# Guardar en session_state por si se necesitan en otras pestañas
st.session_state["df_top"] = df_top
st.session_state["fila_cliente_raw"] = fila_cliente_raw

if df_top is None or df_top.empty:
    st.warning("No se encontraron grasas que cumplan los requisitos actuales.")
    st.stop()

st.success("✅ Recomendaciones generadas correctamente.")

# ---------------------------
# 1. Tabla resumen
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
# 2. Gráfica de ranking (barras)
# ---------------------------
st.subheader("🏆 Ranking (score normalizado)")

fig_bar = plot_ranking(df_top, score_col="score_norm")
st.pyplot(fig_bar)

# ---------------------------
# 3. Radar chart para una grasa específica
# ---------------------------
st.subheader("📡 Comparación detalle vs perfil ideal")

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