"""
app.py — Interfaz web interactiva del proyecto «Predicción de precios de casas»
==============================================================================

Permite introducir las características de una vivienda y ver el precio
estimado por el modelo entrenado en el notebook `prediccion_precios_casas.ipynb`.

IMPORTANTE: antes de ejecutar esta app debes ejecutar el notebook completo una
vez, porque de ahí sale el archivo `modelo_precio_casas.joblib` (el pipeline
completo: preprocesador + mejor modelo) y, opcionalmente, `reporte_metricas.csv`.

Ejecución:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RUTA_MODELO = BASE_DIR / "modelo_precio_casas.joblib"
RUTA_METRICAS = BASE_DIR / "reporte_metricas.csv"

ZONAS = ["Centro", "Norte", "Poniente", "Sur", "Oriente", "Periferia"]
SIST_MONEDA = "USD"

st.set_page_config(
    page_title="Predicción de Precios de Casas",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .bloque-precio {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 18px;
        padding: 28px 32px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(0, 0, 0, .25);
        margin: 18px 0 6px 0;
    }
    .bloque-precio h2 { margin: 0; font-size: 1.05rem; font-weight: 500; opacity: .85; }
    .bloque-precio h1 { margin: 6px 0 0 0; font-size: 2.6rem; font-weight: 800; letter-spacing: .5px; }
    .bloque-precio p  { margin: 8px 0 0 0; font-size: .85rem; opacity: .75; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Carga del modelo (cacheada para no releer el disco en cada re-render)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el modelo entrenado...")
def cargar_modelo(ruta: Path):
    return joblib.load(ruta)


if not RUTA_MODELO.exists():
    st.error(
        "No se encontró `modelo_precio_casas.joblib` junto a `app.py`.\n\n"
        "**Solución:** abre y ejecuta de principio a fin el notebook "
        "`prediccion_precios_casas.ipynb` (sección 7, «Exportación del modelo»). "
        "Eso generará el archivo del modelo en la carpeta del proyecto y esta "
        "app funcionará."
    )
    st.stop()

modelo = cargar_modelo(RUTA_MODELO)

# Métricas opcionales exportadas por el notebook
mae_mejor = None
df_metricas = None
if RUTA_METRICAS.exists():
    df_metricas = pd.read_csv(RUTA_METRICAS)
    mae_mejor = float(df_metricas["MAE"].min())

# ---------------------------------------------------------------------------
# Barra lateral: contexto del proyecto
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏠 Proyecto IA")
    st.markdown(
        """
        **Predicción de precios de casas**

        Modelo de regresión tabular entrenado con scikit-learn.
        Estima el precio continuo de una vivienda en USD a partir de
        sus características.

        **Variables de entrada**
        - Área construida (m²)
        - Habitaciones y baños
        - Año de construcción
        - Garaje y jardín (Sí/No)
        - Ubicación (zona)

        **Pipeline**
        Imputación → escalado/one-hot → modelo.
        El preprocesamiento está *dentro* del pipeline exportado,
        por lo que la app nunca aplica transformaciones a mano.
        """
    )
    st.caption("Proyecto universitario de Inteligencia Artificial · Python + scikit-learn + Streamlit")

# ---------------------------------------------------------------------------
# Encabezado principal
# ---------------------------------------------------------------------------
st.title("Predicción de precios de casas 🏡")
st.markdown(
    "Introduce las características de una vivienda y obtén una **estimación del "
    "precio** generada por el modelo entrenado en el notebook del proyecto."
)
st.divider()

# ---------------------------------------------------------------------------
# Formulario de entrada
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    area_m2 = st.slider("Área construida (m²)", min_value=45.0, max_value=480.0, value=160.0, step=1.0,
                        help="Metros cuadrados útiles de la vivienda.")
    habitaciones = st.slider("Habitaciones", min_value=1, max_value=8, value=3, step=1)
    banos = st.slider("Baños", min_value=1.0, max_value=5.0, value=2.0, step=0.5,
                      help="Se admiten medios baños (1.5, 2.5, ...).")
    anio_construccion = st.number_input("Año de construcción", min_value=1970, max_value=2025, value=2005, step=1)

with col2:
    ubicacion = st.selectbox("Ubicación (zona)", ZONAS, index=0,
                             help="Zona urbana usada por el modelo (codificada con One-Hot).")
    garaje = st.radio("¿Tiene garaje?", ["Sí", "No"], horizontal=True)
    jardin = st.radio("¿Tiene jardín?", ["Sí", "No"], horizontal=True)
    st.info(f"Antigüedad estimada: **{2025 - int(anio_construccion)} años**", icon="📅")

# El DataFrame DEBE tener exactamente las mismas columnas y nombres que en el entrenamiento
entradas = pd.DataFrame(
    [{
        "area_m2": float(area_m2),
        "habitaciones": int(habitaciones),
        "banos": float(banos),
        "anio_construccion": float(anio_construccion),
        "garaje": garaje,
        "jardin": jardin,
        "ubicacion": ubicacion,
    }]
)

predecir = st.button("🔮 Predecir precio", type="primary", use_container_width=True)

if predecir:
    with st.spinner("Estimando el precio..."):
        precio_predicho = float(modelo.predict(entradas)[0])

    rango_txt = f"± ${mae_mejor:,.0f} (MAE del mejor modelo)" if mae_mejor else "sin rango disponible"
    st.markdown(
        f"""
        <div class="bloque-precio">
            <h2>Precio estimado de la vivienda</h2>
            <h1>${precio_predicho:,.0f} <span style="font-size:1rem">{SIST_MONEDA}</span></h1>
            <p>Error medio del modelo: {rango_txt}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Área", f"{area_m2:,.0f} m²")
    c2.metric("Precio / m²", f"${precio_predicho / area_m2:,.0f}")
    c3.metric("Zona", ubicacion)

    with st.expander("¿Qué datos se enviaron al modelo?"):
        st.dataframe(entradas, use_container_width=True)

# ---------------------------------------------------------------------------
# Sección inferior: métricas reales exportadas por el notebook
# ---------------------------------------------------------------------------
with st.expander("📊 Métricas de los modelos (RMSE · MAE · R²)"):
    if df_metricas is not None:
        st.dataframe(
            df_metricas.style.format(
                {"RMSE": "${:,.0f}", "MAE": "${:,.0f}", "R²": "{:.4f}"}
            ),
            use_container_width=True,
        )
        st.caption(
            "RMSE penaliza más los errores grandes · MAE es el error absoluto medio · "
            "R² es la proporción de varianza explicada (1 = ajuste perfecto)."
        )
    else:
        st.write("Ejecuta el notebook para generar `reporte_metricas.csv`.")

with st.expander("⚙️ ¿Cómo funciona la predicción?"):
    st.markdown(
        """
        1. **Imputación:** los valores nulos se rellenan con la mediana (numéricas)
           o la moda (categóricas), ajustada *solo* con los datos de entrenamiento.
        2. **Escalado:** `StandardScaler` para las variables numéricas (necesario en
           Ridge/Lasso; inofensivo en árboles).
        3. **Codificación:** `OneHotEncoder` para la ubicación y binarias `Sí/No`.
        4. **Modelo:** el mejor de la comparativa del notebook (p. ej. Random Forest),
           seleccionado por menor RMSE en el conjunto de prueba.

        Todo viaja dentro de un único `Pipeline` serializado con `joblib`,
        por eso la app no repite ninguna transformación manualmente.
        """
    )

st.divider()
st.caption("Los precios son estimaciones estadísticas, no una tasación oficial. Datos sintéticos o de Kaggle según se configure el notebook.")
