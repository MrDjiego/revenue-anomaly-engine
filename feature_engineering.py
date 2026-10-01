import pandas as pd
import numpy as np
from typing import List

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae características temporales de la fecha.
    
    Justificación: Modelos como LightGBM no interpretan fechas en formato string/datetime.
    Necesitamos desglosar la estacionalidad en variables numéricas explícitas.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Extraemos componentes básicos
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)
    
    # Variables de calendario de negocio (afectan presupuestos y cierres de ventas)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    return df

def create_lag_and_rolling_features(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Crea variables rezagadas (lags) y promedios móviles.
    
    Justificación de Negocio: En SaaS y E-commerce, el ciclo de ventas suele tener
    una fuerte estacionalidad de 7 días. El 'lag_7' compara el día actual con el mismo
    día de la semana pasada. La desviación estándar móvil ('rolling_std_7') ayudará 
    enormemente al Isolation Forest a detectar volatilidad anormal.
    """
    df = df.copy()
    # Aseguramos el orden cronológico estricto para no mezclar datos pasados/futuros
    df = df.sort_values('date').reset_index(drop=True)
    
    for col in target_cols:
        # Lags directos: Valor exacto de hace 1 día y 7 días
        df[f'{col}_lag_1'] = df[col].shift(1)
        df[f'{col}_lag_7'] = df[col].shift(7)
        
        # Ventanas móviles: Capturan la tendencia a corto plazo
        # MUY IMPORTANTE: Usamos shift(1) antes del rolling para evitar "Data Leakage" 
        # (fuga de información), asegurando que no usamos datos de "hoy" para predecir "hoy".
        df[f'{col}_rolling_mean_7'] = df[col].shift(1).rolling(window=7).mean()
        df[f'{col}_rolling_std_7'] = df[col].shift(1).rolling(window=7).std()
        
    return df

def main():
    """Pipeline de ejecución de la Fase 2."""
    print("Cargando datos sintéticos de la Fase 1...")
    input_path = 'synthetic_business_data.csv'
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: No se encontró {input_path}. Ejecuta la Fase 1 primero.")
        return

    print("Generando variables de tiempo...")
    df_features = create_time_features(df)
    
    print("Generando lags y rolling windows...")
    # Seleccionamos las variables clave de negocio para crear sus históricos
    target_columns = ['visits', 'conversion_rate', 'revenue']
    df_features = create_lag_and_rolling_features(df_features, target_columns)
    
    # Al hacer un lag de 7 días, los primeros 7 registros del dataset tendrán NaN.
    # En producción es mejor eliminarlos para entrenar con datos limpios.
    df_features = df_features.dropna().reset_index(drop=True)
    
    output_path = 'engineered_business_data.csv'
    df_features.to_csv(output_path, index=False)
    
    print(f"✅ Ingeniería de características completada con éxito.")
    print(f"Datos guardados en: {output_path}")
    print(f"Nuevas dimensiones del dataset: {df_features.shape}")
    print("\nColumnas listas para entrenamiento:")
    print(df_features.columns.tolist())

if __name__ == "__main__":
    main()