import streamlit as st

st.set_page_config(
    page_title="Recomendador de Grasas Interlub",
    layout="wide"
)
st.image('images/interlub2.png')
st.title("Recomendador de Grasas")

st.markdown("""
### Bienvenido al Recomendador de Grasas Interlub

Esta aplicación te ayuda a:

- **Seleccionar la grasa más adecuada** para una condición de operación específica.
- **Explorar escenarios de “qué pasaría si…”** modificando variables críticas de una grasa real.

Para ello combinamos:

- Rango de **temperatura de servicio** (mínima y máxima)
- **Nivel de carga mecánica** y severidad de la aplicación
- **Presencia de agua / lavado** y ambiente térmico
- Propiedades **reológicas y tribológicas** de las grasas Interlub
- Un **perfil objetivo en espacio latente** construido a partir del cuestionario
- Modelos de **regresión lineal** entrenados sobre datos reales y datos sintéticos generados con un **VAE** para simular escenarios       
""")

st.markdown('---')

st.markdown('### 🔁 Flujo de uso de la app')


with st.expander('**1️⃣ Definir parámetros del cliente**'):
    st.write('''
        En la pestaña **Parámetros del cliente**:
        - Respondes el **cuestionario de condiciones de operación** (ambiente térmico, cargas, presencia de agua, velocidades, etc.).
        - Indicas la **temperatura mínima y máxima de operación**.
        - Defines si existe **presencia de agua / lavado** o ambientes agresivos.
        - Seleccionas el **nivel de carga mecánica / severidad del servicio**.
        - Eliges cuántas **recomendaciones** quieres ver.
        - Escribes una **descripción libre del caso** para análisis de texto.
        - Finalmente presionas **Guardar parámetros**, lo que genera el **perfil objetivo** y/o guarda el texto descriptivo sobre el lubricante.
    ''')

with st.expander('**2️⃣ Obtener recomendaciones de grasas Interlub**'):
    st.write('''
        En la pestaña **Recomendador**:
        
        - Primero eliges **cómo quieres que el sistema recomiende** usando uno de los tres botones:
            - **Formulario** → usa únicamente las respuestas del cuestionario.
            - **Texto** → usa únicamente la descripción libre que escribiste.
            - **Mixto** → combina formulario + texto.
        
        - Dependiendo de tu elección, el sistema construye un **perfil ideal del cliente**:
            - A partir de tus respuestas del formulario.
            - A partir del texto (similitud entre tu descripción y las fichas técnicas).
            - O una mezcla de ambos.
        
        - Después, compara ese perfil con todas las grasas Interlub y calcula qué tan bien
          se ajusta cada una considerando:
            - Temperaturas de operación.
            - Nivel de carga.
            - Presencia de agua o lavado.
            - Perfil técnico general.
            - (Si estás en modo Texto/Mixto) similitud entre tu descripción y la descripción de cada grasa.
        
        - Con esto genera un **score** y produce:
            - Una **tabla con las mejores opciones**.
            - Un **ranking de recomendadas**.
            - **Gráficas tipo radar** para comparar:
                - Tu perfil ideal vs una grasa seleccionada.
                - Grasas recomendadas entre sí (modo texto/mixto).
        
        - La grasa con mayor score se guarda y se usa como
          opción principal en el **Simulador de escenarios**.
    ''')

with st.expander('**3️⃣ Simular escenarios con el regresor (Regresión lineal + VAE)**'):
    st.write('''
        En la pestaña **Simulador de escenarios**:
        - Se cargan los datos históricos de grasas y los **modelos de regresión** entrenados sobre:
            - Datos reales de laboratorio.
            - Datos sintéticos generados con un **VAE** (para enriquecer el espacio de posibles combinaciones).
        - Seleccionas una **grasa real del inventario** por su `codigoGrasa`:
            - Si existe una **mejor_grasa** desde el recomendador, se propone primero como opción.
        - Eliges una **variable crítica a modificar** (por ejemplo, viscosidad del aceite base, punto de gota, desgaste 4 bolas, etc.).
        - Definís un **delta de cambio** para esa variable (por ejemplo, subir o bajar la temperatura o la viscosidad).
        - Ajustas el **factor de ajuste del modelo**:
            - `0`  → cambios muy suaves.
            - `1`  → aplicación completa del ajuste pronosticado por el modelo.
        - Al presionar **🚀 Simular escenario**:
            - El modelo estima **cómo deberían ajustarse las demás propiedades** de la grasa para ser consistentes con ese cambio.
            - Se muestra una **tabla comparativa** (valores originales vs escenario simulado).
            - Se despliega una **gráfica de impacto por variable**, para visualizar qué tanto cambió cada propiedad.
        - Este módulo está pensado para apoyar conversaciones de ingeniería tipo:
            - *“¿Qué pasaría si cambiamos la viscosidad / punto de gota / estabilidad mecánica?”*
            - Sin salirnos de la **coherencia estadística** aprendida a partir de los datos.
    ''')