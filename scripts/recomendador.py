import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from math import pi

# -----------------------------
# 0. Cargar datos y constantes
# -----------------------------

RUTA_RAW_INTERLUB   = "data/datos_grasas_Tec_limpio.csv"
RUTA_MODEL_INTERLUB = "data/datos_grasas_Interlub_limpios_v2.csv"
RUTA_MODEL_COMP     = "data/competidores_preprocesados.csv"  # hoy casi no se usa


def preparar_df_grasas_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols_drop = [
        "idDatosGrasas", "codigoGrasa", "Registro NSF", "categoria",
        "subtitulo", "descripcion", "beneficios", "aplicaciones",
        "Grado NLGI Consistencia",
    ]
    cols_drop = [c for c in cols_drop if c in df.columns]
    df = df.drop(columns=cols_drop)

    cols_num = [
        "Viscosidad del Aceite Base a 40°C. cSt",
        "Penetración de Cono a 25°C, 0.1mm",
        "Punto de Gota, °C",
        "Estabilidad Mecánica, %",
        "Punto de Soldadura Cuatro Bolas, kgf",
        "Desgaste Cuatro Bolas, mm",
        "Indice de Carga-Desgaste",
        "Carga Timken Ok, lb",
        "Resistencia al Lavado por Agua a 80°C, %",
        "Factor de Velocidad",
        "Temperatura de Servicio °C, min",
        "Temperatura de Servicio °C, max",
    ]
    for col in cols_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# Cargar dataframes globales (para usar en las funciones de abajo)
df_interlub_raw   = preparar_df_grasas_raw(pd.read_csv(RUTA_RAW_INTERLUB, encoding='utf-8'))
df_interlub_model = pd.read_csv(RUTA_MODEL_INTERLUB, encoding='utf-8')
df_comp_model     = pd.read_csv(RUTA_MODEL_COMP, encoding='utf-8')   # por ahora casi no lo usamos

feature_cols = df_interlub_model.columns.tolist()

# Grupos de features (en espacio RAW)
TEMP_FEATURES_RAW = [
    "Temperatura de Servicio °C, min",
    "Temperatura de Servicio °C, max",
    "Punto de Gota, °C",
]

CARGA_FEATURES_RAW = [
    "Punto de Soldadura Cuatro Bolas, kgf",
    "Carga Timken Ok, lb",
    "Indice de Carga-Desgaste",
]

AGUA_FEATURES_RAW = [
    "Resistencia al Lavado por Agua a 80°C, %",
]

VISC_FEATURES_RAW = [
    "Viscosidad del Aceite Base a 40°C. cSt",
]

WEIGHTS = {
    "global": 0.4,
    "temp":   0.25,
    "carga":  0.2,
    "agua":   0.1,
    "visc":   0.05,
}
w_sum = sum(WEIGHTS.values())
WEIGHTS = {k: v / w_sum for k, v in WEIGHTS.items()}


# ---------------------------------------
# 1. Filtros y construcción de fila ideal
# ---------------------------------------

def filtrar_por_requisitos_raw(df_raw: pd.DataFrame, req: dict) -> pd.DataFrame:
    df_f = df_raw.copy()

    if req.get("T_min") is not None:
        df_f = df_f[df_f["Temperatura de Servicio °C, min"] <= req["T_min"]]

    if req.get("T_max") is not None:
        df_f = df_f[df_f["Temperatura de Servicio °C, max"] >= req["T_max"]]

    if req.get("ambiente_agua", False):
        if "Resistencia al Lavado por Agua a 80°C, %" in df_f.columns:
            umbral = df_f["Resistencia al Lavado por Agua a 80°C, %"].quantile(0.75)
            df_f = df_f[df_f["Resistencia al Lavado por Agua a 80°C, %"] <= umbral]

    carga = req.get("carga", None)
    if carga in ["media", "alta", "extrema"]:
        if "Punto de Soldadura Cuatro Bolas, kgf" in df_f.columns:
            q_map = {"media": 0.4, "alta": 0.6, "extrema": 0.8}
            q = q_map[carga]
            umbral_4b = df_f["Punto de Soldadura Cuatro Bolas, kgf"].quantile(q)
            df_f = df_f[df_f["Punto de Soldadura Cuatro Bolas, kgf"] >= umbral_4b]

    return df_f


def construir_fila_cliente_raw(df_raw: pd.DataFrame, req: dict) -> pd.DataFrame:
    fila = df_raw.mean(numeric_only=True)

    if req.get("T_min") is not None:
        fila["Temperatura de Servicio °C, min"] = req["T_min"]
    if req.get("T_max") is not None:
        fila["Temperatura de Servicio °C, max"] = req["T_max"]

    if req.get("carga") in ["alta", "extrema"]:
        if "Punto de Soldadura Cuatro Bolas, kgf" in df_raw.columns:
            fila["Punto de Soldadura Cuatro Bolas, kgf"] = df_raw["Punto de Soldadura Cuatro Bolas, kgf"].quantile(0.9)
        if "Carga Timken Ok, lb" in df_raw.columns:
            fila["Carga Timken Ok, lb"] = df_raw["Carga Timken Ok, lb"].quantile(0.9)

    if req.get("ambiente_agua", False):
        if "Resistencia al Lavado por Agua a 80°C, %" in df_raw.columns:
            fila["Resistencia al Lavado por Agua a 80°C, %"] = df_raw["Resistencia al Lavado por Agua a 80°C, %"].quantile(0.1)

    return fila.to_frame().T


# -----------------------------------------------------
# 2. Mapa RAW → modelo y cálculo de similitudes parciales
# -----------------------------------------------------

def construir_mapa_raw_a_model(feature_cols):
    raw_to_model = {}
    for col in feature_cols:
        if col.startswith("num__"):
            base = col.replace("num__", "")
            raw_to_model[base] = col

    def get_indices(raw_list):
        names = []
        for raw_name in raw_list:
            model_name = raw_to_model.get(raw_name, None)
            if model_name is not None and model_name in feature_cols:
                names.append(model_name)
        return [feature_cols.index(n) for n in names]

    idx_temp  = get_indices(TEMP_FEATURES_RAW)
    idx_carga = get_indices(CARGA_FEATURES_RAW)
    idx_agua  = get_indices(AGUA_FEATURES_RAW)
    idx_visc  = get_indices(VISC_FEATURES_RAW)

    idx_groups = {
        "temp": idx_temp,
        "carga": idx_carga,
        "agua": idx_agua,
        "visc": idx_visc,
    }
    return raw_to_model, idx_groups


RAW_TO_MODEL, IDX_GROUPS = construir_mapa_raw_a_model(feature_cols)


def mapear_raw_a_preprocesado(
    fila_raw: pd.Series,
    df_model: pd.DataFrame,
    raw_to_model: dict
) -> pd.DataFrame:
    fila_model = df_model.mean().copy()

    for raw_name, model_name in raw_to_model.items():
        if raw_name in fila_raw.index and model_name in df_model.columns:
            fila_model[model_name] = fila_raw[raw_name]

    return fila_model.to_frame().T


def sim_cos_component(X_ideal, X_cands, idxs):
    if not idxs:
        return None
    return cosine_similarity(X_ideal[:, idxs], X_cands[:, idxs])[0]


def construir_scores(
    X_ideal: np.ndarray,
    X_cands: np.ndarray,
    idx_groups: dict,
    weights: dict
):
    sim_global = cosine_similarity(X_ideal, X_cands)[0]

    sim_temp  = sim_cos_component(X_ideal, X_cands, idx_groups["temp"])
    sim_carga = sim_cos_component(X_ideal, X_cands, idx_groups["carga"])
    sim_agua  = sim_cos_component(X_ideal, X_cands, idx_groups["agua"])
    sim_visc  = sim_cos_component(X_ideal, X_cands, idx_groups["visc"])

    sims = {
        "global": sim_global,
        "temp":   sim_temp  if sim_temp  is not None else np.zeros_like(sim_global),
        "carga":  sim_carga if sim_carga is not None else np.zeros_like(sim_global),
        "agua":   sim_agua  if sim_agua  is not None else np.zeros_like(sim_global),
        "visc":   sim_visc  if sim_visc  is not None else np.zeros_like(sim_global),
    }

    score = (
        weights["global"] * sims["global"]
        + weights["temp"] * sims["temp"]
        + weights["carga"] * sims["carga"]
        + weights["agua"] * sims["agua"]
        + weights["visc"] * sims["visc"]
    )

    return sims, score


# -------------------------------
# 3. Recomendador “pro”
# -------------------------------

def recomendar_interlub_pro(
    req: dict,
    df_raw: pd.DataFrame = df_interlub_raw,
    df_model: pd.DataFrame = df_interlub_model,
    feature_cols_model = feature_cols,
    raw_to_model: dict = RAW_TO_MODEL,
    idx_groups: dict = IDX_GROUPS,
    weights: dict = WEIGHTS,
    top_n: int = 5,
):
    df_cands_raw = filtrar_por_requisitos_raw(df_raw, req)
    if df_cands_raw.empty:
        print("⚠️ No hay grasas que cumplan los requisitos.")
        return None, None

    idx_cands = df_cands_raw.index
    df_cands_model = df_model.iloc[idx_cands]

    fila_cliente_raw = construir_fila_cliente_raw(df_raw, req)

    fila_cliente_model = mapear_raw_a_preprocesado(
        fila_cliente_raw.iloc[0], df_model, raw_to_model
    )

    X_ideal = fila_cliente_model[feature_cols_model].values
    X_cands = df_cands_model[feature_cols_model].values

    sims, score_final = construir_scores(
        X_ideal, X_cands, idx_groups, weights
    )

    df_out = df_cands_raw.copy()
    df_out["sim_global"] = sims["global"]
    df_out["sim_temp"]   = sims["temp"]
    df_out["sim_carga"]  = sims["carga"]
    df_out["sim_agua"]   = sims["agua"]
    df_out["sim_visc"]   = sims["visc"]
    df_out["score"]      = score_final   # lo conservamos para depurar, pero ya no ordena

    numeric_cols_radar = [
        "Temperatura de Servicio °C, min",
        "Temperatura de Servicio °C, max",
        "Resistencia al Lavado por Agua a 80°C, %",
        "Punto de Soldadura Cuatro Bolas, kgf",
        "Carga Timken Ok, lb",
    ]

    # Solo columnas que existan en ambos
    numeric_cols_radar = [
        c for c in numeric_cols_radar
        if c in df_out.columns and c in fila_cliente_raw.columns
    ]

    if numeric_cols_radar:
        # Normalización 0–1 global (todas las grasas)
        mins = df_interlub_raw[numeric_cols_radar].min()
        maxs = df_interlub_raw[numeric_cols_radar].max()

        # Ideal como Serie (no DataFrame)
        ideal_norm = (
            (fila_cliente_raw[numeric_cols_radar].iloc[0] - mins)
            / (maxs - mins + 1e-9)
        )
        prods_norm = (df_out[numeric_cols_radar] - mins) / (maxs - mins + 1e-9)

        # Distancia euclidiana promedio en el espacio radar
        diffs = prods_norm.sub(ideal_norm, axis=1)
        dist_radar = np.sqrt((diffs ** 2).mean(axis=1))

        # ----- Escala ABSOLUTA para el score -----
        # Define hasta qué distancia consideras "ya muy diferente"
        max_dist = 0.8  # puedes ajustar 0.6–1.0 según veas

        dist_clipped = dist_radar.clip(0.0, max_dist)
        sim_radar = 1.0 - dist_clipped / max_dist   # 1 = igual, 0 = muy lejos

        df_out["score_norm"] = sim_radar * 100.0
    else:
        df_out["score_norm"] = 50.0

    # Ordenar por score_norm (lo que realmente ve el cliente en el radar)
    df_out = df_out.sort_values("score_norm", ascending=False)
    df_top = df_out.head(top_n).copy()
    # Explicaciones por producto
    explicaciones = []
    for idx, row in df_top.iterrows():
        partes = []

        if req.get("T_min") is not None:
            diff_min = row["Temperatura de Servicio °C, min"] - req["T_min"]
            if diff_min <= 0:
                partes.append(f"Temp min: ✔️ {row['Temperatura de Servicio °C, min']}°C ≤ {req['T_min']}°C")
            else:
                partes.append(f"Temp min: ⚠️ {row['Temperatura de Servicio °C, min']}°C > {req['T_min']}°C")

        if req.get("T_max") is not None:
            diff_max = row["Temperatura de Servicio °C, max"] - req["T_max"]
            if diff_max >= 0:
                partes.append(f"Temp max: ✔️ {row['Temperatura de Servicio °C, max']}°C ≥ {req['T_max']}°C")
            else:
                partes.append(f"Temp max: ⚠️ {row['Temperatura de Servicio °C, max']}°C < {req['T_max']}°C")

        if req.get("ambiente_agua", False) and "Resistencia al Lavado por Agua a 80°C, %" in df_raw.columns:
            lavado_pct = df_raw["Resistencia al Lavado por Agua a 80°C, %"].rank(pct=True)[idx]
            if lavado_pct <= 0.25:
                partes.append("Lavado: 💧 excelente (muy resistente)")
            elif lavado_pct <= 0.5:
                partes.append("Lavado: bueno")
            else:
                partes.append("Lavado: regular/alto")

        if "Punto de Soldadura Cuatro Bolas, kgf" in df_raw.columns:
            carga_pct = df_raw["Punto de Soldadura Cuatro Bolas, kgf"].rank(pct=True)[idx]
            if carga_pct >= 0.75:
                partes.append("Carga: 🏋️ excelente (alto punto 4 bolas)")
            elif carga_pct >= 0.5:
                partes.append("Carga: buena")
            else:
                partes.append("Carga: baja")

        explicaciones.append(" | ".join(partes))

    df_top["explicacion_score"] = explicaciones

    return df_top, fila_cliente_raw


# -------------------------------
# 4. Gráficas (para notebook / Streamlit)
# -------------------------------

def plot_ranking(df_top, score_col="score_norm", title="Top recomendaciones Interlub"):
    df_tmp = df_top.reset_index().rename(columns={"index": "Grasa", score_col: "Score"})

    chart = (
        alt.Chart(df_tmp)
        .mark_bar(cornerRadius=5)   # bordes redondeados
        .encode(
            x=alt.X("Grasa:N", sort='-y'),
            y=alt.Y("Score:Q", title="Score (%)"),
            tooltip=["Grasa", "Score"]
        )
        .properties(
            title=title,
            width=600,
            height=350
        )
    )

    # Añadir etiquetas encima de cada barra
    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-2
    ).encode(
        text=alt.Text("Score:Q", format=".1f")
    )

    return chart + text


def plot_radar_profile(
    df_all: pd.DataFrame,
    fila_ideal: pd.Series,
    fila_producto: pd.Series,
    numeric_cols,
    title="Perfil radar",
):
    labels = list(numeric_cols)

    # Valores
    vals_ideal = fila_ideal[labels].values.astype(float)
    vals_prod  = fila_producto[labels].values.astype(float)

    # Normalización 0–1
    mins = df_all[labels].min().values.astype(float)
    maxs = df_all[labels].max().values.astype(float)

    vals_ideal_norm = (vals_ideal - mins) / (maxs - mins + 1e-9)
    vals_prod_norm  = (vals_prod  - mins) / (maxs - mins + 1e-9)

    fig = go.Figure()

    # Ideal cliente
    fig.add_trace(
        go.Scatterpolar(
            r=vals_ideal_norm,
            theta=labels,
            fill="toself",
            name="Ideal cliente",
            line=dict(width=3, color="#4C78A8"),             # azul
            fillcolor="rgba(76, 120, 168, 0.45)",
            opacity=0.9,
        )
    )

    # Producto
    fig.add_trace(
        go.Scatterpolar(
            r=vals_prod_norm,
            theta=labels,
            fill="toself",
            name="Producto",
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
            bgcolor="rgba(0,0,0,0)",  # transparente, respeta tema de Streamlit
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".1f",
                tickfont=dict(color="black", size=16),     # números claros
                gridcolor="rgba(200,200,200,0.25)",
                linecolor="rgba(200,200,200,0.4)",
            ),
            angularaxis=dict(
                tickfont=dict(color="black", size=16),     # nombres de variables claros
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
        height=650,   # más grande
    )

    return fig