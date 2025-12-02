# app.regresion.py

import numpy as np
import pandas as pd
import streamlit as st
from scripts import regresor

st.title("Simulador de escenarios de lubricantes (Regresión lineal + VAE)")
st.write(
    """
    Selecciona una grasa real del inventario, elige una variable crítica a modificar
    y observa cómo se ajustan las demás propiedades según los modelos lineales
    entrenados sobre datos reales y sintéticos.
    """
)

# 1) Cargar datos y modelos
datos_grasas_Tec = regresor.cargar_datos()
modelos, variables_criticas = regresor.preparar_modelos(datos_grasas_Tec)

# ---------------- Parametros ----------------
st.header("🔧 Parámetros del escenario")

# Seleccionar grasa por codigoGrasa
codigos = list(datos_grasas_Tec["codigoGrasa"].unique())

# Intentar preseleccionar la mejor grasa proveniente del recomendador
codigo_preseleccionado = None

if "mejor_grasa" in st.session_state:
    mejor = st.session_state["mejor_grasa"]

    # Caso 1: viene como "Grasa_123" (índice del df_interlub_raw / df_top)
    if isinstance(mejor, str) and mejor.startswith("Grasa_"):
        try:
            idx = int(mejor.split("_", 1)[1])
            if idx in datos_grasas_Tec.index:
                codigo_preseleccionado = datos_grasas_Tec.loc[idx, "codigoGrasa"]
        except Exception:
            codigo_preseleccionado = None
    # Caso 2: viene directamente como codigoGrasa
    else:
        if mejor in codigos:
            codigo_preseleccionado = mejor

# Si logramos obtener un código válido, lo movemos al inicio de la lista
if codigo_preseleccionado in codigos:
    codigos.remove(codigo_preseleccionado)
    codigos.insert(0, codigo_preseleccionado)

codigo_sel = st.selectbox("Selecciona una grasa:", options=codigos)

if "mejor_grasa" in st.session_state and codigo_preseleccionado is not None:
    st.caption("La grasa preseleccionada es la mejor grasa según el recomendador.")
else:
    st.caption("Aún no hay una mejor grasa guardada.")

fila_grasa = datos_grasas_Tec[datos_grasas_Tec["codigoGrasa"] == codigo_sel].iloc[0]
grasa_real_vars = fila_grasa[variables_criticas]

# Variable a modificar
variable_cambiada = st.selectbox("Variable a modificar:", options=variables_criticas)

# Rango de delta (heurística simple)
serie_var = datos_grasas_Tec[variable_cambiada].replace(-99, np.nan)
vmin, vmax = float(serie_var.min()), float(serie_var.max())
rango = vmax - vmin if vmax > vmin else 1.0

if "Temperatura" in variable_cambiada or "°C" in variable_cambiada:
    delta_min, delta_max = -20.0, 20.0
    delta_step = 1.0
elif 'Desgaste Cuatro Bolas, mm' in variable_cambiada:
    delta_min, delta_max = -0.13, 0.13
    delta_step = 0.01
else:
    delta_min, delta_max = -0.5 * rango, 0.5 * rango
    delta_step = max(rango / 40.0, 0.1)

delta = st.slider(
    "Cambio (delta) a aplicar",
    min_value=float(delta_min),
    max_value=float(delta_max),
    value=0.0,
    step=float(delta_step)
)

factor_ajuste = st.slider(
    "Factor de ajuste del modelo (0 = muy suave, 1 = completo)",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.1,
)

simular = st.button("🚀 Simular escenario")

# ---------------- Contenido principal ----------------

st.subheader("Grasa real seleccionada")
st.write(f"**codigoGrasa:** `{codigo_sel}`")
st.dataframe(grasa_real_vars.to_frame("Valor original").T)

if simular:
    resumen = regresor.simular_cambio_grasa(
        grasa_real=grasa_real_vars,
        modelos=modelos,
        variable_cambiada=variable_cambiada,
        delta=delta,
        factor_ajuste=factor_ajuste,
    )

    st.subheader("Resumen del escenario")
    st.dataframe(resumen.style.format("{:.2f}"))

    st.subheader("Impacto en las variables")
    fig = regresor.plot_cambio_variables(resumen, variable_cambiada)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        "Ajusta los parámetros en la barra lateral y presiona **🚀 Simular escenario** "
        "para ver el impacto de los cambios."
    )
