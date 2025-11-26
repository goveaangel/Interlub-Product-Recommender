import streamlit as st

st.set_page_config(
    page_title="Recomendador de Grasas Interlub",
    layout="wide"
)

st.title("🧪 Recomendador de Grasas Interlub")

st.markdown("""
### Bienvenido al Recomendador de Grasas Interlub

Esta aplicación ayuda a **seleccionar la grasa más adecuada** para una condición de operación
específica, combinando:

- Rango de temperatura de servicio
- Nivel de carga mecánica
- Presencia de agua / lavado
- Propiedades reológicas y tribológicas de las grasas Interlub
- (Opcional) Comparación contra productos de competidores
            
""")

st.markdown('---')

st.markdown('### 🔁 Flujo de uso de la app')

with st.expander('**1️⃣ Definir parámetros del cliente**'):
   st.write('''
        En la pestaña **Parametros del cliente**:
        - Indicas la **temperatura mínima y máxima de operación**.
        - Definís si hay **presencia de agua / lavado**.
        - Seleccionas el **nivel de carga mecánica**.
        - Eliges cuántas recomendaciones quieres ver.
        - Guardas los parámetros para usarlos en las demás pestañas.
            ''')

with st.expander('**2️⃣ Obtener recomendaciones de grasas Interlub** '):
   st.write('''
        En la pestaña **Recomendador**:
        - Se usan los parámetros guardados para construir un **perfil ideal**.
        - El modelo calcula **similitudes parciales** (temperatura, carga, agua, viscosidad).
        - Genera un **score global normalizado (0–100)** y muestra:
        - Tabla resumen
        - Ranking de grasas
        - Gráfica radar cliente vs grasa seleccionada
            ''')

with st.expander('**3️⃣ Comparar con productos de competidores** '):
   st.write('''
        En la pestaña **Comparacion clientes**:
        - Con los mismos parámetros del cliente, se contrastan:
        - Grasas Interlub recomendadas
        - Productos equivalentes de competidores
        - Se identifican **cuáles competidores se alinean mejor** y dónde Interlub ofrece ventajas.
            ''')