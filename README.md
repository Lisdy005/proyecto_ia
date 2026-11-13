# Predicción de Precios de Casas 🏠

Proyecto universitario de **Inteligencia Artificial**: regresión tabular para estimar el **precio continuo** de una vivienda a partir de sus características (área, habitaciones, baños, año de construcción, garaje, jardín y ubicación).

## Resultado obtenido (respaldo sintético, semilla 42)

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest** 🏆 | **$30,945** | **$24,506** | **0.936** |
| Ridge (CV) | $34,485 | $27,871 | 0.921 |
| Lasso (CV) | $34,486 | $27,871 | 0.921 |
| Regresión Lineal | $34,486 | $27,872 | 0.921 |

MAPE ≈ 10.2 % · Verificación de recarga `joblib`: predicciones idénticas a las del pipeline original.

## Estructura del proyecto

```
proyecto_ia/
├── prediccion_precios_casas.ipynb   # Notebook principal (ejecutar de arriba hacia abajo)
├── app.py                           # Interfaz web en Streamlit (usa el modelo exportado)
├── requirements.txt                  # Dependencias con versiones fijadas
├── README.md                        # Este archivo
├── data/
│   └── casas.csv                    # (Opcional) CSV real; si no existe, el notebook genera uno sintético
├── modelo_precio_casas.joblib       # Generado por la sección 7 del notebook (pipeline completo)
├── reporte_metricas.csv             # Generado por el notebook (tabla comparativa)
└── mapa_precios.html                # Generado por la sección 6 (mapa interactivo Folium)
```

## Inicio rápido (orden obligatorio)

```bash
# 1) Entorno virtual limpio
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) Instalar dependencias
pip install -r requirements.txt

# 3) Ejecutar el notebook completo (genera modelo_precio_casas.joblib y reporte_metricas.csv)
jupyter notebook prediccion_precios_casas.ipynb
#    → Kernel > Restart & Run All
#    o, sin navegador:
#    jupyter nbconvert --to notebook --execute --inplace prediccion_precios_casas.ipynb

# 4) Lanzar la app web
streamlit run app.py               # → http://localhost:8501
```

> **Orden:** primero el notebook, después Streamlit. Si la app no encuentra `modelo_precio_casas.joblib`, mostrará instrucciones en pantalla.

## Usar un dataset real de Kaggle

1. Descarga un CSV de viviendas (p. ej. *Housing Prices Dataset*, `yasserh/housing-prices-dataset`) y **renombra sus columnas** al esquema: `area_m2, habitaciones, banos, anio_construccion, garaje, jardin, ubicacion, precio` (tipos en la sección 1.2 del notebook).
2. Guárdalo como `data/casas.csv`.
3. Vuelve a ejecutar el notebook: detectará el archivo real y **sustituirá al respaldo sintético** automáticamente, sin tocar código.

## Qué incluye el notebook

- Prototipo de 10 filas en un DataFrame de pandas (tipos mixtos int/float/categórico).
- Carga con `pd.read_csv` + generador sintético de respaldo (n=3 000, semilla 42) con nulos y duplicados realistas.
- EDA, tratamiento de nulos, codificación one-hot y split 80/20 con **pipeline anti-fuga** (imputación y escalado ajustados solo con train).
- 4 modelos: Regresión Lineal, Random Forest, Ridge y Lasso con `GridSearchCV` + validación cruzada 5-fold.
- Tabla comparativa RMSE / MAE / R² con interpretación y guía de defensa.
- Análisis de residuos del mejor modelo (residuos vs. predichos, distribución, reales vs. predichos, Q-Q).
- Informe de importancia de características: MDI, permutación y coeficientes estandarizados.
- Mapa interactivo de precios por zona (Folium, coordenadas coherentes con CDMX).
- Exportación del pipeline completo con `joblib` + app en Streamlit.

## Stack

Python 3.11 · pandas · scikit-learn · matplotlib · seaborn · Folium · Streamlit · joblib · Jupyter

---

*Proyecto universitario de IA — Predicción de precios de casas. Los precios son estimaciones estadísticas, no una tasación oficial.*
