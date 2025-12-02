import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import altair as alt

#---------------
# CARGA DE DATOS
#---------------
data = pd.read_csv('data/datos_grasas_tec_limpio.csv', encoding='utf-8')

# columnas
data_col = [
    "Aceite Base",
    "Espesante",
    "categoria",
    "subtitulo",
    "descripcion",
    "beneficios",
    "aplicaciones",
    "color",
    "textura",
]

def construir_texto_completo(df):
    # Asegurar que todas las columnas de texto existan (o crearlas vacías)
    columnas_presentes = []
    for col in data_col:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
            columnas_presentes.append(col)

    # Construimos el texto completo uniendo las columnas que sí están
    df["texto_completo"] = (
        df[columnas_presentes].agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()
    )

    return df

data_grasas = construir_texto_completo(data)

#--------------------------
# MODELO + MATRIZ SIMILITUD
#--------------------------

# Vectorizador TF-IDF
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),  # unigrams + bigrams
    min_df=1,
)

# Matriz TF-IDF de todas las grasas
tfidf_matrix = tfidf.fit_transform(data_grasas["texto_completo"])

#-------------------
# FUNCIONES PUBLICAS
#-------------------

def catalogo_texto():

    return data_grasas


def recomendar_por_texto(texto, top_n: int = 5):
    """
    Recomienda grasas a partir de un texto libre (descripción / necesidad).

    Parameters
    ----------
    texto : str
        Texto libre que describe la necesidad del cliente.
    top_n : int
        Número de recomendaciones a devolver.

    Returns
    -------
    DataFrame
        Sub-dataframe con las columnas originales de df_grasas + columna 'sim_texto'
        (similitud coseno respecto al texto ingresado).
    """


    # 1. Vectorizar texto del usuario
    texto_vec = tfidf.transform([texto])

    # 2. Similitud con todas las grasas
    sim_scores = cosine_similarity(texto_vec, tfidf_matrix).flatten()

    # 3. Crear DF completo con TODAS las grasas + similitud
    df_all = data_grasas.copy()
    df_all["sim_texto"] = sim_scores

    # 4. Normalización GLOBAL (usa todas las grasas)
    s_min = df_all["sim_texto"].min()
    s_max = df_all["sim_texto"].max()

    if s_max > s_min:
        df_all["score_norm_texto"] = 100 * (df_all["sim_texto"] - s_min) / (s_max - s_min)
    else:
        df_all["score_norm_texto"] = 50.0

    # 5. Ordenar por similitud global antes del top_n
    df_all = df_all.sort_values("sim_texto", ascending=False)

    # 6. Ahora sí recortar al top_n
    df_top = df_all.head(top_n).copy()

    return df_top

# -------------------------------
# GRAFICAS
# -------------------------------
import plotly.graph_objects as go  # asegúrate de tener esto arriba del archivo

def plot_radar_texto(
    df_all: pd.DataFrame,
    fila_ideal: pd.Series,
    fila_producto: pd.Series,
    numeric_cols,
    title="Radar (modo texto)",
):
    """
    Radar para el recomendador de TEXTO.

    - 'Ideal' = una grasa de referencia (por ejemplo, la top 1 de df_top_texto)
    - 'Producto' = grasa seleccionada para comparar

    Parameters
    ----------
    df_all : DataFrame
        DataFrame numérico completo (por ejemplo df_interlub_raw).
    fila_ideal : Series
        Fila (grasa) que se toma como referencia "ideal".
    fila_producto : Series
        Fila (grasa) seleccionada por el usuario.
    numeric_cols : list
        Columnas numéricas a graficar en el radar.
    title : str
        Título del gráfico.
    """

    labels = list(numeric_cols)

    # 1) Extraer valores en el mismo orden de labels
    vals_ideal = fila_ideal[labels].values.astype(float)
    vals_prod  = fila_producto[labels].values.astype(float)

    # 2) Normalización 0–1 usando TODO df_all
    mins = df_all[labels].min().values.astype(float)
    maxs = df_all[labels].max().values.astype(float)

    vals_ideal_norm = (vals_ideal - mins) / (maxs - mins + 1e-9)
    vals_prod_norm  = (vals_prod  - mins) / (maxs - mins + 1e-9)

    # 3) Construir figura
    fig = go.Figure()

    # Ideal texto
    fig.add_trace(
        go.Scatterpolar(
            r=vals_ideal_norm,
            theta=labels,
            fill="toself",
            name="Ideal recomendador",
            line=dict(width=3, color="#4C78A8"),             # azul
            fillcolor="rgba(76, 120, 168, 0.45)",
            opacity=0.9,
        )
    )

    # Producto seleccionado
    fig.add_trace(
        go.Scatterpolar(
            r=vals_prod_norm,
            theta=labels,
            fill="toself",
            name="Grasa seleccionada",
            line=dict(width=3, color="#F58518"),             # naranja
            fillcolor="rgba(245, 133, 24, 0.45)",
            opacity=0.9,
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=22, color='black')
        ),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".1f",
                tickfont=dict(color="black", size=16),
                gridcolor="rgba(200,200,200,0.25)",
                linecolor="rgba(200,200,200,0.4)",
            ),
            angularaxis=dict(
                tickfont=dict(color="black", size=16),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(color="black", size=16),
        ),
        margin=dict(l=40, r=40, t=80, b=80),
        height=650,
    )

    return fig