import streamlit as st

st.title("1️⃣ Parámetros del cliente")

st.markdown("""
En esta sección vamos a definir las **condiciones de operación** del cliente.  
""")

# ---------------------------
# Layout en la página 
# ---------------------------

with st.form("form_parametros_cliente"):
    st.subheader("📌 Condiciones de operación")

    col1, col2 = st.columns(2)

    with col1:
        T_min = st.slider(
            "Temperatura mínima requerida (°C)",
            min_value=-60,
            max_value=50,
            value=-10,
            step=5,
        )

        top_n = st.slider(
            "Número de recomendaciones a mostrar",
            min_value=3,
            max_value=10,
            value=5,
        )

        ambiente_agua = st.checkbox(
            "Ambiente con presencia de agua (lavado / humedad)",
            value=True,
        )

    with col2:
        T_max = st.slider(
            "Temperatura máxima requerida (°C)",
            min_value=60,
            max_value=250,
            value=130,
            step=10,
        )

        op_carga = st.selectbox(
            "Nivel de carga mecánica",
            options=["No especificar", "media", "alta", "extrema"],
            index=2,
        )

    guardar = st.form_submit_button("💾 Guardar parámetros")

if guardar:
    carga = None if op_carga == "No especificar" else op_carga

    req = {
        "T_min": T_min,
        "T_max": T_max,
        "ambiente_agua": ambiente_agua,
        "carga": carga,
    }

    st.session_state["req"] = req
    st.session_state["top_n"] = top_n

    st.success("✅ Parámetros guardados correctamente.")
else:
    if "req" in st.session_state:
        st.info("Ya hay parámetros guardados. Puedes modificarlos y volver a guardar.")
    else:
        st.info("Aún no hay parámetros guardados. Completa el formulario y presiona **Guardar parámetros**.")

with st.form('Texto_vector_counterizer'):
    st.subheader('📝 Descripcion completa del lubricante')

    descripcion_cliente = st.text_area(
        label="Escribe aquí la descripción completa del cliente:",
        placeholder="Ejemplo: Cliente del sector automotriz que requiere lubricantes con alta resistencia al agua...",
        height=150
    )

    guardar2 =st.form_submit_button('💾 Guardar descripción')

if guardar2:
    if descripcion_cliente.strip() != "":

        st.session_state["descripcion_cliente"] = descripcion_cliente
        st.success("✅ Parámetros guardados correctamente.")
    else:
        st.error("🚨 No puedes guardar un texto vacío. Por favor escribe una descripción.")
else:
    # Mostrar mensaje informativo SOLO si aún no se ha guardado
    if "descripcion_cliente" not in st.session_state:
        st.info("Aún no hay un texto guardado. Completa la descripción y presiona **Guardar descripción**.")
    else:
        st.info("Ya hay una descripción guardada. Puedes modificarla y volver a guardar.")


