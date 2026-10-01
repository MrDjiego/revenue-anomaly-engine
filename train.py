import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, classification_report
from typing import Tuple, List

def train_anomaly_detector(df: pd.DataFrame) -> Tuple[IsolationForest, pd.DataFrame]:
    """
    Entrena un Isolation Forest para detectar anomalías multivariadas.
    
    Justificación de Negocio (Wikimedia / Seguridad): Un pico de tráfico no es 
    necesariamente malo (podría ser viralidad). Pero un pico de tráfico 
    con una caída drástica en la conversión es un claro indicador de tráfico 
    basura o un ataque de bots. El Isolation Forest es excelente para encontrar 
    estas relaciones no lineales y atípicas sin necesidad de etiquetas previas.
    """
    print("--- Entrenando Modelo de Detección de Anomalías ---")
    
    # Seleccionamos variables clave para el comportamiento del usuario
    features = ['visits', 'conversion_rate', 'visits_rolling_std_7', 'conversion_rate_rolling_std_7']
    X = df[features]
    
    # contamination=0.01 asume que ~1% de los días tendrán eventos anómalos severos
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    # El modelo devuelve -1 para anomalías y 1 para datos normales
    df['anomaly_prediction'] = model.fit_predict(X)
    df['is_anomaly_pred'] = np.where(df['anomaly_prediction'] == -1, 1, 0)
    
    # Evaluamos contra nuestra "Ground Truth" inyectada en la Fase 1
    print("\nReporte de Detección (Ground Truth vs Predicción):")
    print(classification_report(df['is_anomaly'], df['is_anomaly_pred']))
    
    return model, df

def train_forecasting_model(df: pd.DataFrame) -> lgb.LGBMRegressor:
    """
    Entrena un modelo LightGBM para pronosticar los ingresos (Revenue).
    
    Justificación de Negocio (Vonage / Revenue Ops): Predecir los ingresos con
    precisión permite ajustar presupuestos de marketing y detectar si el cierre 
    de mes alcanzará la cuota. Usamos LightGBM porque maneja excepcionalmente bien
    las variables categóricas (días de la semana) y valores atípicos.
    """
    print("\n--- Entrenando Modelo Predictivo de Ingresos ---")
    
    # Preparamos los datos ordenados por fecha (CRÍTICO en Time Series)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Definimos features (excluimos la fecha y las variables objetivo actuales)
    drop_cols = ['date', 'is_anomaly', 'anomaly_type', 'sales', 'revenue', 'anomaly_prediction', 'is_anomaly_pred']
    features = [col for col in df.columns if col not in drop_cols]
    target = 'revenue'
    
    X = df[features]
    y = df[target]
    
    # Split cronológico: Últimos 60 días para validación (Out-of-Time Validation)
    test_size = 60
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        random_state=42,
        importance_type='gain'
    )
    
    # Entrenamos el modelo
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    
    # Evaluación
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    
    print(f"\nMétricas en Test (Últimos {test_size} días):")
    print(f"MAE (Error Absoluto Medio): ${mae:,.2f}")
    print(f"MAPE (Error Porcentual): {mape*100:.2f}%")
    
    # Importancia de variables
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Variables más predictivas:")
    print(importance.head())
    
    # Guardamos la lista de features exactas requeridas para la API en la Fase 4
    # model.feature_names_in_ = features 
    
    return model

def main():
    """Pipeline de ejecución de la Fase 3."""
    os.makedirs('models', exist_ok=True)
    
    print("Cargando datos con ingeniería de características...")
    df = pd.read_csv('engineered_business_data.csv')
    
    # 1. Entrenamos Anomaly Detection
    iso_forest_model, df_evaluated = train_anomaly_detector(df)
    
    # 2. Entrenamos Forecasting
    lgbm_model = train_forecasting_model(df_evaluated)
    
    # 3. Serialización de modelos para producción
    iso_path = 'models/isolation_forest.pkl'
    lgb_path = 'models/lgbm_forecaster.pkl'
    
    joblib.dump(iso_forest_model, iso_path)
    joblib.dump(lgbm_model, lgb_path)
    
    print(f"\n✅ Modelos guardados en '{iso_path}' y '{lgb_path}' listos para la API.")

if __name__ == "__main__":
    main()