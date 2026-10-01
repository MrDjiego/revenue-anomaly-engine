from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import logging
import os

# Configuración de logs para monitoreo en producción
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Motor Predictivo de Ingresos y Anomalías",
    description="API REST para predecir ingresos diarios y detectar anomalías en tráfico web/conversión.",
    version="1.0.0"
)

# 1. Obtenemos la ruta absoluta de la carpeta donde está este script (api.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 2. Inicializamos las variables
iso_model = None
lgbm_model = None

# 3. Cargamos los modelos
try:
    iso_model = joblib.load(os.path.join(MODEL_DIR, 'isolation_forest.pkl'))
    lgbm_model = joblib.load(os.path.join(MODEL_DIR, 'lgbm_forecaster.pkl'))
    logger.info("✅ Modelos cargados exitosamente en memoria.")
except Exception as e:
    logger.error(f"❌ Error CRÍTICO al cargar los modelos: {str(e)}")

class DailyMetrics(BaseModel):
    """
    Esquema de validación estricta para el payload de entrada.
    Justificación MLOps: Validar los tipos de datos antes de que lleguen al modelo
    evita el 90% de los errores silenciosos en producción.
    """
    visits: float = Field(..., description="Visitas totales del día")
    conversion_rate: float = Field(..., description="Tasa de conversión del día (ej. 0.025 para 2.5%)")
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)
    is_month_start: int = Field(..., ge=0, le=1)
    is_month_end: int = Field(..., ge=0, le=1)
    visits_lag_1: float
    visits_lag_7: float
    visits_rolling_mean_7: float
    visits_rolling_std_7: float
    conversion_rate_lag_1: float
    conversion_rate_lag_7: float
    conversion_rate_rolling_mean_7: float
    conversion_rate_rolling_std_7: float
    revenue_lag_1: float
    revenue_lag_7: float
    revenue_rolling_mean_7: float
    revenue_rolling_std_7: float

@app.post("/detect", tags=["Seguridad / Operaciones"])
def detect_anomaly(data: DailyMetrics):
    """
    Endpoint para Wikimedia / Infraestructura:
    Evalúa si el tráfico y conversión del día actual representan una anomalía
    (ej. Bot attack, caída de pasarela de pagos).
    """
    try:
        # Extraemos solo las variables que Isolation Forest necesita
        features = [[
            data.visits, 
            data.conversion_rate, 
            data.visits_rolling_std_7, 
            data.conversion_rate_rolling_std_7
        ]]
        
        # El modelo retorna -1 para anomalía, 1 para normal
        prediction = iso_model.predict(features)[0]
        is_anomaly = bool(prediction == -1)
        
        return {
            "status": "success",
            "is_anomaly": is_anomaly,
            "message": "Tráfico anómalo detectado. Revisar logs de seguridad." if is_anomaly else "Tráfico dentro de los parámetros normales."
        }
    except Exception as e:
        logger.error(f"Error en /detect: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la anomalía.")

@app.post("/predict", tags=["Negocio / Finanzas"])
def predict_revenue(data: DailyMetrics):
    """
    Endpoint para Vonage / Revenue Ops:
    Pronostica los ingresos esperados basados en el tráfico, conversión y 
    patrones históricos recientes.
    """
    try:
        # Compatibilidad para Pydantic V1 y V2
        payload_dict = data.model_dump() if hasattr(data, 'model_dump') else data.dict()
        
        # Convertimos el payload a DataFrame
        df_input = pd.DataFrame([payload_dict])
        
        prediction = lgbm_model.predict(df_input)[0]
        
        return {
            "status": "success",
            "predicted_revenue_usd": round(float(prediction), 2)
        }
    except Exception as e:
        logger.error(f"Error en /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno al generar el pronóstico: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "online", "model_version": "1.0.0"}
