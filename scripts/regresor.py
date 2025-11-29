#  Regresor

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 0. FUNCIÓN VAE (BÁSICA / DE PRUEBA)
# ============================================================

def simular_datos_vae(df, n_nuevos=1000, ruido=0.05):
    """
    SIMULADOR SIMPLE (placeholder) PARA PROBAR LA APP.
    ------------------------------------------------------------------
    - Recibe un DataFrame numérico df.
    - Genera n_nuevos muestras:
        - hace resampling de filas reales
        - añade ruido gaussiano proporcional (ruido * desviación estándar)
    - Devuelve un DataFrame con las mismas columnas.
    
    ⚠️ IMPORTANTE:
    - Cuando tengas tu función real del VAE, reemplaza esta función por:
        from scripts.lo_que_sea import simular_datos_vae
    - Pero respeta la firma: (df, n_nuevos) -> DataFrame
    """
    df_num = df.copy()
    cols = df_num.columns

    # re-sample de filas reales
    base = df_num.sample(n=min(len(df_num), n_nuevos), replace=True, random_state=42).reset_index(drop=True)

    # si quieres exactamente n_nuevos
    if len(base) < n_nuevos:
        extra = df_num.sample(n=n_nuevos - len(base), replace=True, random_state=123).reset_index(drop=True)
        base = pd.concat([base, extra], ignore_index=True)

    # agregar ruido
    std = df_num.std()
    ruido_mat = np.random.normal(loc=0.0, scale=(std * ruido).values, size=base.shape)
    sim = base.values + ruido_mat

    sim_df = pd.DataFrame(sim, columns=cols)
    return sim_df


# ============================================================
# 1. FUNCIONES BASE DE REGRESIÓN
# ============================================================

def variables_regresion(df, variable_objetivo, variables_predictoras):
    """
    Prepara X e y para una regresión lineal:
    - Filtra filas donde alguna variable tenga el valor -99.
    - Imputa NaN con la mediana de cada columna relevante.
    """
    df_filtrado = df.copy()
    columnas = variables_predictoras + [variable_objetivo]

    # Quitar -99
    for var in columnas:
        df_filtrado = df_filtrado[df_filtrado[var] != -99]

    # Imputar NaN con mediana columna por columna
    for col in columnas:
        mediana = df_filtrado[col].median()
        df_filtrado[col] = df_filtrado[col].fillna(mediana)

    X = df_filtrado[variables_predictoras]
    y = df_filtrado[variable_objetivo]
    return X, y, variables_predictoras


def entrenar_modelos_multivariables(df_expanded, variables_criticas):
    """
    Entrena un modelo LinearRegression por cada variable crítica usando
    las demás variables críticas como predictoras.

    Devuelve:
    modelos[variable_objetivo] = {
        "modelo": LinearRegression(),
        "scaler": StandardScaler(),
        "predictoras": [...],
        "metricas": {...}
    }
    """
    modelos = {}

    for variable_objetivo in variables_criticas:
        variables_predictoras = [v for v in variables_criticas if v != variable_objetivo]

        X, y, features = variables_regresion(df_expanded, variable_objetivo, variables_predictoras)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        modelo = LinearRegression()
        modelo.fit(X_train_scaled, y_train)

        modelos[variable_objetivo] = {
            "modelo": modelo,
            "scaler": scaler,
            "predictoras": features,
            "metricas": {
                "mse_train": mean_squared_error(y_train, modelo.predict(X_train_scaled)),
                "mse_test": mean_squared_error(y_test, modelo.predict(X_test_scaled)),
                "r2_train": r2_score(y_train, modelo.predict(X_train_scaled)),
                "r2_test": r2_score(y_test, modelo.predict(X_test_scaled)),
            },
        }

    return modelos


def simular_cambio_grasa(
    grasa_real,
    modelos,
    variable_cambiada,
    delta,
    factor_ajuste=0.5,
):
    """
    Simula el efecto de modificar una variable de una grasa real.

    Parámetros
    ----------
    grasa_real : pd.Series
        Fila real con las variables_criticas.
    modelos : dict
        Diccionario de modelos entrenados (salida de entrenar_modelos_multivariables).
    variable_cambiada : str
        Variable que el usuario quiere modificar.
    delta : float
        Cambio que se aplica (ej. +5 °C).
    factor_ajuste : float
        Entre 0 y 1. Cuánto confiamos en el valor del modelo:
        1.0 = cambio completo, 0.5 = cambio suavizado.

    Devuelve
    --------
    resumen : pd.DataFrame
        Índice = nombres de variables_criticas.
        Columnas = ["valor_original", "valor_modificado_input", "valor_predicho_modelo"].
    """
    if not isinstance(grasa_real, pd.Series):
        grasa_real = grasa_real.squeeze()

    variables = list(modelos.keys())

    resumen = pd.DataFrame(
        index=variables,
        columns=["valor_original", "valor_modificado_input", "valor_predicho_modelo"],
        dtype=float
    )

    # valores originales
    for v in variables:
        resumen.loc[v, "valor_original"] = grasa_real[v]

    # aplicar delta manual
    modificada = grasa_real.copy()
    modificada[variable_cambiada] = modificada[variable_cambiada] + delta

    for v in variables:
        resumen.loc[v, "valor_modificado_input"] = modificada[v]

    # predicción con cada modelo
    for v_obj, info in modelos.items():
        modelo = info["modelo"]
        scaler = info["scaler"]
        preds = info["predictoras"]

        X_input = modificada[preds].values.reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        y_pred = modelo.predict(X_scaled)[0]

        y_original = grasa_real[v_obj]
        y_nuevo = y_original + factor_ajuste * (y_pred - y_original)

        resumen.loc[v_obj, "valor_predicho_modelo"] = y_nuevo

    return resumen


def plot_cambio_variables(resumen: pd.DataFrame, variable_cambiada: str):
    """
    Gráfica de barras (Plotly) comparando:
    - valor_original
    - valor_predicho_modelo
    para cada variable en el índice de `resumen`.
    """
    # Pasar índice a columna
    df_plot = resumen.copy().reset_index().rename(columns={"index": "Variable"})

    # Renombrar columnas para nombres bonitos
    df_plot = df_plot.rename(
        columns={
            "valor_original": "Original",
            "valor_predicho_modelo": "Después (modelo)",
        }
    )

    # Formato largo para Plotly
    df_long = df_plot.melt(
        id_vars="Variable",
        value_vars=["Original", "Después (modelo)"],
        var_name="Serie",         # nuevo nombre
        value_name="Valor",
    )

    fig = px.bar(
        df_long,
        x="Variable",
        y="Valor",
        color="Serie",
        barmode="group",
        text="Valor",
        title=f"Cambio en variables al modificar '{variable_cambiada}'",
        color_discrete_map={
            "Original": "#4C78A8",          # azul
            "Después (modelo)": "#F58518",  # naranja
        },
    )

    # Ajustes estéticos
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:.3f}<extra></extra>",
    )

    fig.update_layout(
        xaxis=dict(
            tickangle=-45,
        ),
        yaxis=dict(
            title="Valor",
            zeroline=True,
            zerolinecolor="rgba(200,200,200,0.4)",
        ),
        legend=dict(
            title="",   # <<< Sin título en la leyenda
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#E0E0E0", size=12),
        ),
        margin=dict(l=40, r=40, t=80, b=120),
        height=700,
    )

    return fig


# ============================================================
# 2. Funciones cacheadas para Streamlit
# ============================================================

import pandas as pd
import warnings
# from scripts.tu_modulo_vae import simular_datos_vae
# from scripts.tu_modulo_modelos import entrenar_modelos_multivariables


def cargar_datos(ruta_csv: str = "data/datos_grasas_Tec_limpio.csv") -> pd.DataFrame:
    """
    Carga el CSV con las grasas reales y limpia las columnas numéricas críticas.
    """
    df = pd.read_csv(ruta_csv, encoding="latin1")

    # Arreglar nombres raros de columnas (Â° → °)
    df = df.rename(columns={
        "Viscosidad del Aceite Base a 40Â°C. cSt": "Viscosidad del Aceite Base a 40°C. cSt",
        "Punto de Gota, Â°C": "Punto de Gota, °C",
        "Temperatura de Servicio Â°C, min": "Temperatura de Servicio °C, min",
        "Temperatura de Servicio Â°C, max": "Temperatura de Servicio °C, max",
    })

    # Columnas numéricas críticas que vas a usar en el VAE y en los modelos
    cols_numericas = [
        "Viscosidad del Aceite Base a 40°C. cSt",
        "Punto de Gota, °C",
        "Punto de Soldadura Cuatro Bolas, kgf",
        "Desgaste Cuatro Bolas, mm",
        "Temperatura de Servicio °C, min",
        "Temperatura de Servicio °C, max",
    ]

    # Asegurar que existan y sean numéricas
    for col in cols_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imputar NaN con la mediana de cada columna crítica
    df[cols_numericas] = df[cols_numericas].fillna(df[cols_numericas].median(numeric_only=True))

    return df


def preparar_modelos(df: pd.DataFrame):
    """
    Prepara variables_criticas, genera datos con VAE y entrena modelos lineales.

    Esta función NO depende de Streamlit (backend puro).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original con las grasas reales.

    Returns
    -------
    modelos : dict o el tipo que devuelva entrenar_modelos_multivariables
        Modelos entrenados.
    variables_criticas : list
        Lista final de variables usadas para entrenar los modelos.
    """
    variables_criticas = [
        "Viscosidad del Aceite Base a 40°C. cSt",
        "Punto de Gota, °C",
        "Punto de Soldadura Cuatro Bolas, kgf",
        "Desgaste Cuatro Bolas, mm",
        "Temperatura de Servicio °C, min",
        "Temperatura de Servicio °C, max",
    ]

    # input al VAE
    df_vae_input = df[variables_criticas].copy()

    expanded_data = simular_datos_vae(df_vae_input, n_nuevos=1000)

    # si el VAE regresa array, le ponemos nombres
    if not isinstance(expanded_data, pd.DataFrame):
        expanded_data = pd.DataFrame(expanded_data, columns=variables_criticas)

    # asegurarnos de que solo usamos columnas válidas
    cols_expanded = expanded_data.columns.tolist()
    cols_comunes = [c for c in variables_criticas if c in cols_expanded]

    if len(cols_comunes) < len(variables_criticas):
        faltan = set(variables_criticas) - set(cols_comunes)
        warnings.warn(
            f"Estas columnas no se encuentran en expanded_data y no se usarán: {faltan}",
            UserWarning,
        )

    variables_criticas = cols_comunes
    expanded_data = expanded_data[variables_criticas]

    modelos = entrenar_modelos_multivariables(expanded_data, variables_criticas)

    return modelos, variables_criticas