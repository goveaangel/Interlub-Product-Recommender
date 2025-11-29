import streamlit as st

from scripts.recomendador import (
    df_interlub_raw,
    recomendar_interlub_pro,
    plot_ranking,
    plot_radar_profile,
)

st.title("2️⃣ Recomendador de grasas Interlub")

st.markdown("""
Usando las respuestas del **cuestionario de parámetros del cliente**, 
el sistema calcula un perfil objetivo (**v2_deseado**) y, con base en éste, 
genera el **ranking de grasas Interlub** que mejor se ajustan al perfil.
""")

# ---------------------------
# Verificar parámetros
# ---------------------------
if "req" not in st.session_state or "top_n" not in st.session_state:
    st.warning(
        "Primero ve a **Parámetros del cliente** para completar el cuestionario y guardar los parámetros."
    )
    st.stop()

req = st.session_state["req"]               # viene de cuestionario_grasas (v2_deseado)
top_n = st.session_state["top_n"]
latent_levels = st.session_state.get("latent_levels", None)

st.subheader("📌 Parámetros en uso (perfil objetivo v2_deseado)")

# Extraer del req el vector objetivo
temp_min_servicio_obj = req["temp_min_servicio_obj"]
temp_max_servicio_obj = req["temp_max_servicio_obj"]
punto_gota_obj = req["punto_gota_obj"]
punto_soldadura_4b_obj = req["punto_soldadura_4b_obj"]
desgaste_4b_obj = req["desgaste_4b_obj"]

# ---------------------------
# Mostrar resumen en métricas
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("T° mínima de servicio (objetivo)", f"{temp_min_servicio_obj:.1f} °C")

with col2:
    st.metric("T° máxima de servicio (objetivo)", f"{temp_max_servicio_obj:.1f} °C")

with col3:
    st.metric("Punto de gota (objetivo)", f"{punto_gota_obj:.1f} °C")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Punto soldadura 4B (objetivo)", f"{punto_soldadura_4b_obj:.1f} kgf")

with col5:
    st.metric("Desgaste 4B (objetivo)", f"{desgaste_4b_obj:.3f} mm")

with col6:
    if latent_levels is not None:
        crit = latent_levels.get("Criticidad_niv", None)
        if crit is not None:
            st.metric("Nivel de criticidad (1–5)", f"{crit}")
        else:
            st.write("Nivel de criticidad: N/D")
    else:
        st.write("Nivel de criticidad: N/D")

st.markdown("---")

# Flag en session_state para saber si ya corrimos el recomendador
if "recom_ejecutado" not in st.session_state:
    st.session_state["recom_ejecutado"] = False

ejecutar = st.button("🔍 Ejecutar recomendador")

if ejecutar:
    # Si se presiona el botón, calculamos nuevamente
    with st.spinner("Calculando recomendaciones..."):
        df_top, fila_cliente_raw = recomendar_interlub_pro(
            req=req,
            top_n=top_n,
        )

    if df_top is None or df_top.empty:
        st.session_state["recom_ejecutado"] = False
        st.warning("No se encontraron grasas que cumplan los requisitos actuales.")
        st.stop()

    # Guardar en session_state
    st.session_state["df_top"] = df_top
    st.session_state["fila_cliente_raw"] = fila_cliente_raw
    st.session_state["recom_ejecutado"] = True
    st.success("✅ Recomendaciones generadas correctamente.")

    if df_top is not None and not df_top.empty:

        index_mejor = df_top.index[0]
        mejor_grasa = f"Grasa_{index_mejor}"
        st.session_state['mejor_grasa'] = mejor_grasa

# Si todavía no se ha ejecutado nunca, mostramos el mensaje y paramos
if not st.session_state["recom_ejecutado"]:
    st.info("Presiona el botón para generar las recomendaciones con estos parámetros.")
    st.stop()

# A partir de aquí, SIEMPRE tomamos los datos desde session_state
df_top = st.session_state["df_top"]
fila_cliente_raw = st.session_state["fila_cliente_raw"]

# ---------------------------
# Llamar al backend
# ---------------------------
with st.spinner("Calculando recomendaciones..."):
    # ⚠️ IMPORTANTE:
    # Aquí `req` ahora contiene el v2_deseado, NO T_min/T_max/agua/carga.
    # Asegúrate de adaptar la función `recomendar_interlub_pro` para que
    # use estas claves:
    #   - 'punto_gota_obj'
    #   - 'punto_soldadura_4b_obj'
    #   - 'desgaste_4b_obj'
    #   - 'temp_min_servicio_obj'
    #   - 'temp_max_servicio_obj'
    df_top, fila_cliente_raw = recomendar_interlub_pro(
        req=req,
        top_n=top_n,
    )

# Guardar en session_state por si se necesitan en otras pestañas
st.session_state["df_top"] = df_top
st.session_state["fila_cliente_raw"] = fila_cliente_raw

if df_top is None or df_top.empty:
    st.warning("No se encontraron grasas que cumplan los requisitos actuales.")
    st.stop()

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

fig_bar = plot_ranking(df_top)
st.altair_chart(fig_bar, use_container_width=True)

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
    df_all=df_interlub_raw,
    fila_ideal=fila_ideal,
    fila_producto=fila_producto,
    numeric_cols=numeric_cols,
    title=f"Radar de Grasa #{idx_opcion}",
)

st.plotly_chart(fig_radar, use_container_width=True)