import streamlit as st
from dataclasses import dataclass
from typing import List, Optional

# -------------------------------------------------------------------
# Dataclasses de dominio (idénticas al módulo original)
# -------------------------------------------------------------------

@dataclass
class QuestionnaireResponse:
    """Respuestas crudas del cuestionario al cliente."""
    # Bloque temperatura
    ambiente_termico: int       # P1 (1-5)
    temp_max_opcion: int        # P2 (1-5)
    arranques_en_frio: int      # P3 (1-5)
    problemas_temp_escurre: bool
    problemas_temp_quemado: bool
    problemas_temp_arranque_duro: bool
    problemas_temp_ninguno: bool

    # Bloque carga y tipo de esfuerzo
    tipo_carga: int             # P5 (1-5)
    arranques_paros: int        # P6 (1-5)
    sobrecargas_frecuencia: int # P7 (1-3)

    # Bloque criticidad y desgaste
    criticidad_equipo: int      # P8 (1-5)
    historial_fallas: int       # P9 (1-3)
    tolerancia_desgaste: int    # P10 (1-5)

    # Bloque industria y aplicación
    tipo_equipo: int            # P11 (1-6, donde 6 = "otro")
    industria: int              # P12 (1-6, donde 6 = "otra")
    cond_agua_lavado: bool
    cond_polvo: bool
    cond_contacto_alimentos: bool
    cond_vibraciones: bool
    cond_sin_especiales: bool


@dataclass
class LatentLevels:
    """Variables latentes internas (1-5)."""
    T_nivel: int          # severidad térmica
    Frio_nivel: int       # importancia de arranque en frío
    Carga_nivel: int      # severidad de carga general
    Choque_nivel: int     # presencia de golpes de carga
    Criticidad_niv: int   # importancia de minimizar desgaste


@dataclass
class V2Vector:
    """Vector objetivo aproximado para v2."""
    punto_gota_obj: float          # °C
    punto_soldadura_4b_obj: float  # kgf
    desgaste_4b_obj: float         # mm
    temp_min_servicio_obj: float   # °C
    temp_max_servicio_obj: float   # °C

    def to_list(self) -> List[float]:
        return [
            self.punto_gota_obj,
            self.punto_soldadura_4b_obj,
            self.desgaste_4b_obj,
            self.temp_min_servicio_obj,
            self.temp_max_servicio_obj,
        ]


# -------------------------------------------------------------------
# 2. Preprocesado: respuestas → niveles internos
# -------------------------------------------------------------------

def calcular_T_nivel(respuestas: QuestionnaireResponse) -> int:
    base = respuestas.temp_max_opcion  # ya es 1-5

    if respuestas.ambiente_termico in (2, 5):  # interior caliente o cerca de hornos
        base += 1

    if respuestas.problemas_temp_escurre or respuestas.problemas_temp_quemado:
        base += 1

    return max(1, min(5, base))


def calcular_Frio_nivel(respuestas: QuestionnaireResponse) -> int:
    base = respuestas.arranques_en_frio  # 1-5

    if respuestas.ambiente_termico == 4:
        base += 1

    if respuestas.problemas_temp_arranque_duro:
        base += 1

    return max(1, min(5, base))


def calcular_Carga_nivel(respuestas: QuestionnaireResponse) -> int:
    base = respuestas.tipo_carga  # 1-5

    if respuestas.sobrecargas_frecuencia == 2:
        base += 1
    elif respuestas.sobrecargas_frecuencia == 3:
        base += 2

    return max(1, min(5, base))


def calcular_Choque_nivel(respuestas: QuestionnaireResponse) -> int:
    nivel = 1

    if respuestas.tipo_carga == 5:
        nivel = 4

    if respuestas.sobrecargas_frecuencia == 2:
        nivel += 1
    elif respuestas.sobrecargas_frecuencia == 3:
        nivel += 2

    if respuestas.arranques_paros >= 3:
        nivel += 1
    if respuestas.arranques_paros >= 5:
        nivel += 1

    return max(1, min(5, nivel))


def calcular_Criticidad_nivel(respuestas: QuestionnaireResponse) -> int:
    if respuestas.historial_fallas == 1:
        hist_factor = 1
    elif respuestas.historial_fallas == 2:
        hist_factor = 3
    else:
        hist_factor = 5

    nivel = round(
        (respuestas.criticidad_equipo +
         respuestas.tolerancia_desgaste +
         hist_factor) / 3
    )

    return max(1, min(5, nivel))


def calcular_latent_levels(respuestas: QuestionnaireResponse) -> LatentLevels:
    T_nivel = calcular_T_nivel(respuestas)
    Frio_nivel = calcular_Frio_nivel(respuestas)
    Carga_nivel = calcular_Carga_nivel(respuestas)
    Choque_nivel = calcular_Choque_nivel(respuestas)
    Criticidad_niv = calcular_Criticidad_nivel(respuestas)

    return LatentLevels(
        T_nivel=T_nivel,
        Frio_nivel=Frio_nivel,
        Carga_nivel=Carga_nivel,
        Choque_nivel=Choque_nivel,
        Criticidad_niv=Criticidad_niv,
    )


# -------------------------------------------------------------------
# 3. Niveles → vector objetivo aproximado v2_deseado
# -------------------------------------------------------------------

def _map_T_nivel_a_Tmax(T_nivel: int, tipo_equipo: int) -> float:
    base_map = {
        1: 70.0,
        2: 90.0,
        3: 115.0,
        4: 145.0,
        5: 180.0,
    }
    tmax = base_map.get(T_nivel, 100.0)

    if tipo_equipo == 2:        # hornos/ventiladores calientes
        tmax += 10.0
    elif tipo_equipo == 1:      # motores eléctricos
        tmax -= 5.0

    return tmax


def _map_Frio_nivel_a_Tmin(Frio_nivel: int, ambiente_termico: int) -> float:
    if Frio_nivel <= 1:
        tmin = -10.0
    elif Frio_nivel == 2:
        tmin = -15.0
    elif Frio_nivel == 3:
        tmin = -20.0
    elif Frio_nivel == 4:
        tmin = -25.0
    else:
        tmin = -30.0

    if ambiente_termico == 4:
        tmin -= 5.0

    return tmin


def _map_carga_y_choque_a_punto_soldadura(Carga_nivel: int, Choque_nivel: int) -> float:
    severidad = (Carga_nivel + Choque_nivel) / 2.0

    if severidad <= 2:
        rango = (160.0, 220.0)
    elif severidad <= 3.5:
        rango = (220.0, 260.0)
    else:
        rango = (260.0, 320.0)

    return sum(rango) / 2.0


def _map_criticidad_a_desgaste_4b(Criticidad_niv: int) -> float:
    if Criticidad_niv <= 2:
        rango = (0.60, 0.70)
    elif Criticidad_niv <= 4:
        rango = (0.50, 0.60)
    else:
        rango = (0.40, 0.50)

    return sum(rango) / 2.0


def _map_Tmax_a_punto_gota(Tmax_obj: float) -> float:
    return Tmax_obj + 40.0


def construir_v2_deseado(
    T_nivel: int,
    Frio_nivel: int,
    Carga_nivel: int,
    Choque_nivel: int,
    Criticidad_niv: int,
    respuestas: QuestionnaireResponse,
) -> V2Vector:
    temp_max_servicio_obj = _map_T_nivel_a_Tmax(T_nivel, respuestas.tipo_equipo)
    temp_min_servicio_obj = _map_Frio_nivel_a_Tmin(Frio_nivel, respuestas.ambiente_termico)

    punto_gota_obj = _map_Tmax_a_punto_gota(temp_max_servicio_obj)
    punto_soldadura_obj = _map_carga_y_choque_a_punto_soldadura(Carga_nivel, Choque_nivel)
    desgaste_4b_obj = _map_criticidad_a_desgaste_4b(Criticidad_niv)

    return V2Vector(
        punto_gota_obj=punto_gota_obj,
        punto_soldadura_4b_obj=punto_soldadura_obj,
        desgaste_4b_obj=desgaste_4b_obj,
        temp_min_servicio_obj=temp_min_servicio_obj,
        temp_max_servicio_obj=temp_max_servicio_obj,
    )


def transformar_respuestas_a_v2(respuestas: QuestionnaireResponse):
    niveles = calcular_latent_levels(respuestas)
    v2_deseado = construir_v2_deseado(
        T_nivel=niveles.T_nivel,
        Frio_nivel=niveles.Frio_nivel,
        Carga_nivel=niveles.Carga_nivel,
        Choque_nivel=niveles.Choque_nivel,
        Criticidad_niv=niveles.Criticidad_niv,
        respuestas=respuestas,
    )
    return niveles, v2_deseado


# -------------------------------------------------------------------
# 4. Página Streamlit: cuestionario del cliente
# -------------------------------------------------------------------

st.title("1️⃣ Parámetros del cliente")

st.markdown("""
En esta sección vamos a definir las **condiciones de operación** del cliente mediante un cuestionario.  
A partir de las respuestas, el sistema calcula variables **latentes** (severidad térmica, carga, choque, criticidad)
y construye un vector objetivo aproximado **v2_deseado** para el recomendador de grasas.
""")

# ---------------------------
# Layout en la página 
# ---------------------------

with st.form("form_parametros_cliente"):
    st.subheader("📌 Condiciones de operación")

    # ------------------------------------------------------------------
    # BLOQUE TEMPERATURA (P1–P4)
    # ------------------------------------------------------------------
    st.markdown("### 🔥 Bloque temperatura")

    opciones_p1 = [
        ("Interior, ambiente fresco (< 40 °C)", 1),
        ("Interior, caliente (40–80 °C)", 2),
        ("Exterior, ambiente normal (–5 a 40 °C)", 3),
        ("Exterior con frío intenso / cámaras de refrigeración", 4),
        ("Muy cerca de fuentes de calor intenso (hornos, secadores, etc.)", 5),
    ]
    p1_sel = st.selectbox(
        "P1. ¿En qué ambiente trabaja principalmente el equipo lubricado con esta grasa?",
        options=opciones_p1,
        index=0,
        format_func=lambda x: x[0],
    )
    ambiente_termico = p1_sel[1]

    opciones_p2 = [
        ("Menos de 60 °C", 1),
        ("60–90 °C", 2),
        ("90–120 °C", 3),
        ("120–160 °C", 4),
        ("Más de 160 °C", 5),
    ]
    p2_sel = st.selectbox(
        "P2. ¿Cuál es la temperatura aproximada más alta que alcanza la zona lubricada?",
        options=opciones_p2,
        index=1,
        format_func=lambda x: x[0],
    )
    temp_max_opcion = p2_sel[1]

    arranques_en_frio = st.slider(
        "P3. Frecuencia de arranque en temperaturas muy frías (< 0 °C) "
        "(1 = nunca, 3 = a veces, 5 = muy frecuente)",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
    )

    st.markdown("**P4. Problemas observados por temperatura con la grasa actual (marca lo que aplique):**")
    col_p4a, col_p4b = st.columns(2)
    with col_p4a:
        problemas_temp_escurre = st.checkbox("La grasa se escurre o gotea cuando el equipo está caliente")
        problemas_temp_quemado = st.checkbox("Se ve humo o 'quemado' de grasa")
    with col_p4b:
        problemas_temp_arranque_duro = st.checkbox("En frío, el equipo arranca 'duro' o con ruido metálico")
        problemas_temp_ninguno = st.checkbox("No hemos tenido problemas de temperatura relevantes")

    # ------------------------------------------------------------------
    # BLOQUE CARGA Y ESFUERZO (P5–P7)
    # ------------------------------------------------------------------
    st.markdown("### ⚙️ Bloque carga y esfuerzo")

    opciones_p5 = [
        ("Carga ligera y constante", 1),
        ("Carga media y constante", 2),
        ("Carga pesada pero relativamente estable", 3),
        ("Cargas muy pesadas y constantes", 4),
        ("Cargas de choque / golpes", 5),
    ]
    p5_sel = st.selectbox(
        "P5. ¿Cómo describiría la carga mecánica que ve este punto?",
        options=opciones_p5,
        index=2,
        format_func=lambda x: x[0],
    )
    tipo_carga = p5_sel[1]

    arranques_paros = st.slider(
        "P6. Frecuencia de arranques y paros "
        "(1 = casi siempre continuo, 3 = arranques/paros diarios, 5 = muchos ciclos por hora)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    opciones_p7 = [
        ("Rara vez o nunca", 1),
        ("Ocasionalmente (atascos, picos de carga)", 2),
        ("Frecuentemente (golpes, atascos, impacto en el proceso)", 3),
    ]
    p7_sel = st.selectbox(
        "P7. ¿Qué tan frecuente es que se presente sobrecarga o golpes de carga?",
        options=opciones_p7,
        index=0,
        format_func=lambda x: x[0],
    )
    sobrecargas_frecuencia = p7_sel[1]

    # ------------------------------------------------------------------
    # BLOQUE CRITICIDAD Y DESGASTE (P8–P10)
    # ------------------------------------------------------------------
    st.markdown("### 🛡️ Bloque criticidad y desgaste")

    criticidad_equipo = st.slider(
        "P8. Criticidad del equipo "
        "(1 = bajo impacto si se detiene, 5 = crítico, se para la línea/planta)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    opciones_p9 = [
        ("No hay registros de fallas", 1),
        ("Una o dos fallas menores", 2),
        ("Fallas frecuentes o graves", 3),
    ]
    p9_sel = st.selectbox(
        "P9. En los últimos 12 meses, ¿ha tenido fallas por desgaste en este punto?",
        options=opciones_p9,
        index=0,
        format_func=lambda x: x[0],
    )
    historial_fallas = p9_sel[1]

    tolerancia_desgaste = st.slider(
        "P10. ¿Qué tan aceptable es un desgaste 'normal' de componentes? "
        "(1 = totalmente aceptable, 5 = no es aceptable)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    # ------------------------------------------------------------------
    # BLOQUE INDUSTRIA Y APLICACIÓN (P11–P13)
    # ------------------------------------------------------------------
    st.markdown("### 🏭 Bloque industria y aplicación")

    opciones_p11 = [
        ("Rodamientos de motores eléctricos", 1),
        ("Rodamientos de hornos o ventiladores calientes", 2),
        ("Engranajes abiertos / coronas dentadas", 3),
        ("Chumaceras de molinos / mezcladoras pesadas", 4),
        ("Cintas transportadoras / rodillos", 5),
        ("Otro", 6),
    ]
    p11_sel = st.selectbox(
        "P11. El punto a lubricar corresponde principalmente a:",
        options=opciones_p11,
        index=0,
        format_func=lambda x: x[0],
    )
    tipo_equipo = p11_sel[1]

    opciones_p12 = [
        ("Alimentos y bebidas", 1),
        ("Acero / metalurgia", 2),
        ("Cemento / minería", 3),
        ("Plásticos / inyección", 4),
        ("Automotriz", 5),
        ("Otra", 6),
    ]
    p12_sel = st.selectbox(
        "P12. Industria principal:",
        options=opciones_p12,
        index=0,
        format_func=lambda x: x[0],
    )
    industria = p12_sel[1]

    st.markdown("**P13. Condiciones especiales (marca lo que aplique):**")
    col_p13a, col_p13b = st.columns(2)
    with col_p13a:
        cond_agua_lavado = st.checkbox("Presencia de agua o lavado frecuente")
        cond_polvo = st.checkbox("Polvo / contaminación sólida intensa")
    with col_p13b:
        cond_contacto_alimentos = st.checkbox("Contacto incidental con alimentos (H1)")
        cond_vibraciones = st.checkbox("Vibraciones fuertes")
    cond_sin_especiales = st.checkbox("Ninguna condición especial relevante")

    # ------------------------------------------------------------------
    # Parámetro general de la app (como en tu ejemplo original)
    # ------------------------------------------------------------------
    top_n = st.slider(
        "Número de recomendaciones a mostrar",
        min_value=3,
        max_value=10,
        value=5,
    )

    guardar = st.form_submit_button("💾 Guardar parámetros")

# ----------------------------------------------------------------------
# Procesamiento y guardado en session_state
# ----------------------------------------------------------------------
if guardar:
    # Normalización P4 (coherencia con "ninguno")
    if problemas_temp_ninguno:
        problemas_temp_escurre = False
        problemas_temp_quemado = False
        problemas_temp_arranque_duro = False
    else:
        if problemas_temp_escurre or problemas_temp_quemado or problemas_temp_arranque_duro:
            problemas_temp_ninguno = False

    # Normalización P13
    if cond_sin_especiales:
        cond_agua_lavado = False
        cond_polvo = False
        cond_contacto_alimentos = False
        cond_vibraciones = False
    else:
        if cond_agua_lavado or cond_polvo or cond_contacto_alimentos or cond_vibraciones:
            cond_sin_especiales = False

    respuestas = QuestionnaireResponse(
        ambiente_termico=ambiente_termico,
        temp_max_opcion=temp_max_opcion,
        arranques_en_frio=arranques_en_frio,
        problemas_temp_escurre=problemas_temp_escurre,
        problemas_temp_quemado=problemas_temp_quemado,
        problemas_temp_arranque_duro=problemas_temp_arranque_duro,
        problemas_temp_ninguno=problemas_temp_ninguno,
        tipo_carga=tipo_carga,
        arranques_paros=arranques_paros,
        sobrecargas_frecuencia=sobrecargas_frecuencia,
        criticidad_equipo=criticidad_equipo,
        historial_fallas=historial_fallas,
        tolerancia_desgaste=tolerancia_desgaste,
        tipo_equipo=tipo_equipo,
        industria=industria,
        cond_agua_lavado=cond_agua_lavado,
        cond_polvo=cond_polvo,
        cond_contacto_alimentos=cond_contacto_alimentos,
        cond_vibraciones=cond_vibraciones,
        cond_sin_especiales=cond_sin_especiales,
    )

    niveles, v2_deseado = transformar_respuestas_a_v2(respuestas)

    # Lo que va a usar el recomendador (puedes ajustar nombres si quieres)
    req = {
        "punto_gota_obj": v2_deseado.punto_gota_obj,
        "punto_soldadura_4b_obj": v2_deseado.punto_soldadura_4b_obj,
        "desgaste_4b_obj": v2_deseado.desgaste_4b_obj,
        "temp_min_servicio_obj": v2_deseado.temp_min_servicio_obj,
        "temp_max_servicio_obj": v2_deseado.temp_max_servicio_obj,
    }

    st.session_state["req"] = req
    st.session_state["top_n"] = top_n
    st.session_state["questionnaire_raw"] = respuestas.__dict__
    st.session_state["latent_levels"] = niveles.__dict__
    st.session_state["v2_deseado"] = v2_deseado.to_list()

    st.success("✅ Parámetros del cuestionario guardados correctamente.")

else:
    if "req" in st.session_state:
        st.info("Ya hay parámetros de cuestionario guardados. Puedes modificarlos y volver a guardar.")
    else:
        st.info("Aún no hay parámetros guardados. Completa el cuestionario y presiona **Guardar parámetros**.")


# -------------------------------------------------------------------
# 5. Texto libre para CountVectorizer (tal cual tu ejemplo)
# -------------------------------------------------------------------

with st.form('Texto_vector_counterizer'):
    st.subheader('📝 Descripción completa del lubricante')

    descripcion_cliente = st.text_area(
        label="Escribe aquí la descripción completa del cliente:",
        placeholder=(
            "Ejemplo: Cliente del sector automotriz que requiere lubricantes con alta "
            "resistencia al agua, cargas de choque y operación a altas temperaturas..."
        ),
        height=150
    )

    guardar2 = st.form_submit_button('💾 Guardar descripción')

if guardar2:
    if descripcion_cliente.strip() != "":
        st.session_state["descripcion_cliente"] = descripcion_cliente
        st.success("✅ Descripción guardada correctamente.")
    else:
        st.error("🚨 No puedes guardar un texto vacío. Por favor escribe una descripción.")
else:
    if "descripcion_cliente" not in st.session_state:
        st.info("Aún no hay un texto guardado. Completa la descripción y presiona **Guardar descripción**.")
    else:
        st.info("Ya hay una descripción guardada. Puedes modificarla y volver a guardar.")