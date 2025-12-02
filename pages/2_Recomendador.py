import streamlit as st

from scripts.recomendador import (
    df_interlub_raw,
    recomendar_interlub_pro,
    plot_ranking,
    plot_radar_profile,
)
from scripts.recomendador_palabras import recomendar_por_texto, plot_radar_texto

st.title("2️⃣ Recomendador de grasas Interlub")

st.markdown("""
El perfil objetivo del cliente se genera a partir de la información capturada en la sección anterior,
ya sea mediante el **formulario**, el **texto descriptivo** o el **modo mixto**.

Con este perfil, el sistema evalúa todas las grasas disponibles y produce un 
**ranking de compatibilidad**, mostrando cuáles formulaciones de Interlub se ajustan mejor a las 
condiciones reales de operación y a las necesidades expresadas por el usuario.
""")

# -------------------------------------------------
# BOTONES EN COLUMNAS
# -------------------------------------------------
if "modo_recomendador" not in st.session_state:
    st.session_state["modo_recomendador"] = None

col1, col2 = st.columns(2)

with col1:
    if st.button("🧮 Formulario"):
        st.session_state["modo_recomendador"] = "Formulario"

with col2:
    if st.button("✍️ Texto"):
        st.session_state["modo_recomendador"] = "Texto"


st.markdown("---")
modo = st.session_state["modo_recomendador"]
st.write(f"**Modo actual:** {modo}")

if st.session_state["modo_recomendador"] == 'Formulario':
    # ---------------------------
    # Verificar parámetros
    # ---------------------------
    if "req_v2" not in st.session_state or "req_pro" not in st.session_state or "top_n" not in st.session_state:
        st.warning(
            "Primero ve a **Parámetros del cliente** para completar el cuestionario y guardar los parámetros."
        )
        st.stop()

    # req_v2 = perfil técnico objetivo (v2) para mostrar métricas
    req_v2 = st.session_state["req_v2"]
    # req_pro = requisitos en formato que entiende recomendar_interlub_pro
    req_pro = st.session_state["req_pro"]
    top_n = st.session_state["top_n"]
    latent_levels = st.session_state.get("latent_levels", None)

    st.subheader("📌 Parámetros en uso (perfil objetivo v2_deseado)")

    # Extraer del req_v2 el vector objetivo
    temp_min_servicio_obj = req_v2["temp_min_servicio_obj"]
    temp_max_servicio_obj = req_v2["temp_max_servicio_obj"]
    punto_gota_obj = req_v2["punto_gota_obj"]
    punto_soldadura_4b_obj = req_v2["punto_soldadura_4b_obj"]
    desgaste_4b_obj = req_v2["desgaste_4b_obj"]

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
                req=req_pro,       # 👈 AQUÍ USAMOS req_pro (T_min, T_max, carga, agua)
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
    
elif st.session_state["modo_recomendador"] == 'Texto':

    # -----------------------------------
    # 0) Verificacion inicial
    # -----------------------------------

    if 'descripcion_cliente' not in st.session_state:
        st.warning("Primero ve a **Parámetros del cliente** para escribir la **descripcion del lubricante** y guardar la descripcion.")
        st.stop()      

    st.subheader("✍️ Recomendador basado solo en texto")

    # -----------------------------------
    # 1) Inicializar cosas en session_state
    # -----------------------------------
    if "df_top_texto" not in st.session_state:
        st.session_state["df_top_texto"] = None

    if "texto_ejecutado" not in st.session_state:
        st.session_state["texto_ejecutado"] = False

    top_n = st.session_state.get("top_n", 5)

    descripcion_cliente = st.session_state['descripcion_cliente']

    # -----------------------------------
    # 2) Input de usuario
    # -----------------------------------

    ejecutar_texto = st.button("🔍 Buscar grasas por texto")

    # -----------------------------------
    # 3) Ejecutar recomendador SOLO si se presiona el botón
    # -----------------------------------
    if ejecutar_texto:
        if descripcion_cliente.strip() == "":
            st.error("Por favor escribe una descripción antes de buscar.")
        else:
            with st.spinner("Analizando texto y buscando grasas similares..."):
                try:
                    df_top_texto = recomendar_por_texto(descripcion_cliente, top_n)

                    if df_top_texto is None or df_top_texto.empty:
                        st.warning("No se encontraron grasas similares para este texto.")
                        st.session_state["df_top_texto"] = None
                        st.session_state["texto_ejecutado"] = False
                            
                    else:
                        # Guardamos resultados en session_state
                        st.session_state["df_top_texto"] = df_top_texto
                        st.session_state["texto_ejecutado"] = True
                        st.success("✅ Recomendaciones por texto generadas correctamente.")

                        index_mejor = df_top_texto.index[0]
                        mejor_grasa = f"Grasa_{index_mejor}"
                        st.session_state['mejor_grasa'] = mejor_grasa

                except Exception as e:
                    st.error(f"❌ Ocurrió un error al generar las recomendaciones por texto: {e}")
                    st.session_state["df_top_texto"] = None
                    st.session_state["texto_ejecutado"] = False

    # -----------------------------------
    # 4) A partir de aquí, SIEMPRE usamos lo guardado en session_state
    #    (esto se ejecuta también cuando mueves los selectboxes)
    # -----------------------------------
    df_top_texto = st.session_state["df_top_texto"]

    if not st.session_state["texto_ejecutado"] or df_top_texto is None:
        st.info("Escribe una descripción y presiona el botón para generar recomendaciones por texto.")
        st.stop()

    # -----------------------------------
    # 5) Tabla resumen
    # -----------------------------------
    st.subheader("📋 Top grasas recomendadas (por texto)")

    cols_texto = [
        "codigoGrasa",
        "categoria",
        "subtitulo",
        "descripcion",
        "sim_texto",
        "score_norm_texto",
    ]
    cols_texto = [c for c in cols_texto if c in df_top_texto.columns]

    st.dataframe(
        df_top_texto[cols_texto].style.format(
            {
                "sim_texto": "{:.3f}",
                "score_norm_texto": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

    # -----------------------------------
    # 6) Ranking (reutilizando plot_ranking)
    # -----------------------------------
    st.subheader("🏆 Ranking por similitud de texto")

    fig_bar_texto = plot_ranking(
        df_top_texto.rename(columns={"score_norm_texto": "score_norm"}),
        score_col="score_norm",
        title="Top grasas (similitud de texto)",
    )
    st.altair_chart(fig_bar_texto, use_container_width=True)

    # ---------------------------
    # 7) Radar chart para una grasa específica (modo texto)
    # ---------------------------
    st.subheader("📡 Comparación dentro del grupo recomendado")

    # Asegurarnos de que df_top_texto está disponible (ya calculado)
    df_top_texto = st.session_state.get("df_top_texto", df_top_texto)

    # 👉 Grasa de referencia (Ideal) = la mejor del ranking por texto (primer renglón)
    idx_ideal_texto = df_top_texto.index[0]

    st.caption(f"Comparando contra la grasa de referencia (Ideal texto): **Grasa #{idx_ideal_texto}**")

    # Opciones de índice: solo las grasas recomendadas por texto
    idx_opcion_texto = st.selectbox(
        "Selecciona la grasa recomendada para ver su radar:",
        options=list(df_top_texto.index),
        index=0,
        format_func=lambda idx: f"Grasa #{idx}",
        key="idx_radar_texto",
    )

    # Filas en el espacio RAW (numérico)
    fila_producto_texto = df_interlub_raw.loc[idx_opcion_texto]
    fila_ideal_texto    = df_interlub_raw.loc[idx_ideal_texto]

    # Variables numéricas que queremos ver
    numeric_cols = [
        "Temperatura de Servicio °C, min",
        "Temperatura de Servicio °C, max",
        "Resistencia al Lavado por Agua a 80°C, %",
        "Punto de Soldadura Cuatro Bolas, kgf",
        "Carga Timken Ok, lb",
    ]
    numeric_cols = [c for c in numeric_cols if c in df_interlub_raw.columns]

    # Usamos la nueva función de radar para TEXTO
    fig_radar_texto = plot_radar_texto(
        df_all=df_interlub_raw,
        fila_ideal=fila_ideal_texto,
        fila_producto=fila_producto_texto,
        numeric_cols=numeric_cols,
        title=f"Radar (modo texto) – Grasa #{idx_opcion_texto}",
    )

    st.plotly_chart(fig_radar_texto, use_container_width=True)

else: 
    st.info('Selecciona un tipo de recomendador')

