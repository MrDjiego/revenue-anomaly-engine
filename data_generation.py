import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Tuple

def generate_base_metrics(start_date: date, days: int = 730) -> pd.DataFrame:
    """
    Genera la base del tráfico web y conversiones con tendencia y estacionalidad.
    """
    np.random.seed(42)  # Fomentamos la reproducibilidad para el desarrollo
    
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # 1. Tráfico Base (Simulando crecimiento continuo de un SaaS/E-commerce)
    base_traffic = 5000
    trend = np.linspace(0, 3000, days)  # Crecimiento lineal a lo largo de 2 años
    
    # Estacionalidad semanal (picos en días laborables para B2B SaaS, caídas en fin de semana)
    weekly_seasonality = np.array([500 if d.weekday() < 5 else -800 for d in dates])
    
    # Ruido aleatorio para realismo estadístico
    noise_traffic = np.random.normal(0, 200, days)
    
    total_visits = base_traffic + trend + weekly_seasonality + noise_traffic
    total_visits = np.clip(total_visits, 1000, None).astype(int)
    
    # 2. Tasa de Conversión Base (~2.5% promedio, métrica estándar en E-commerce/SaaS)
    base_conv_rate = 0.025
    noise_conv = np.random.normal(0, 0.002, days)
    conversion_rate = np.clip(base_conv_rate + noise_conv, 0.005, 0.10)
    
    df = pd.DataFrame({
        'date': dates,
        'visits': total_visits,
        'conversion_rate': conversion_rate
    })
    
    return df

def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inyecta eventos anómalos de negocio para evaluar modelos de detección posteriores.
    Incluye etiquetas (is_anomaly) para facilitar la validación del modelo Isolation Forest.
    """
    df = df.copy()
    df['is_anomaly'] = 0
    df['anomaly_type'] = 'none'
    
    # Caso 1: Bot Attack / DDoS (Interés para Wikimedia/Seguridad)
    # Pico masivo de tráfico, pero la tasa de conversión se desploma (tráfico basura).
    bot_days = [100, 450, 600]
    for day in bot_days:
        df.loc[day, 'visits'] *= 4.5
        df.loc[day, 'conversion_rate'] *= 0.1
        df.loc[day, 'is_anomaly'] = 1
        df.loc[day, 'anomaly_type'] = 'bot_attack'
        
    # Caso 2: Caída de la Pasarela de Pago (Interés para Vonage/Revenue Operations)
    # Tráfico normal, pero la conversión cae a casi cero por un fallo técnico.
    checkout_crash_days = [250, 520]
    for day in checkout_crash_days:
        df.loc[day, 'conversion_rate'] *= 0.05
        df.loc[day, 'is_anomaly'] = 1
        df.loc[day, 'anomaly_type'] = 'checkout_crash'
        
    # Caso 3: Campaña Viral Exitosa o Mega-Deal B2B (Anomalía Positiva)
    # Aumento de tráfico con excelente tasa de conversión.
    viral_days = [300, 680]
    for day in viral_days:
        df.loc[day, 'visits'] *= 2.5
        df.loc[day, 'conversion_rate'] *= 1.5
        df.loc[day, 'is_anomaly'] = 1
        df.loc[day, 'anomaly_type'] = 'viral_campaign'
        
    return df

def calculate_revenue(df: pd.DataFrame, aov: float = 120.0) -> pd.DataFrame:
    """
    Calcula las ventas y los ingresos basados en el tráfico y la conversión.
    AOV: Average Order Value (Valor Promedio del Pedido).
    """
    df['sales'] = (df['visits'] * df['conversion_rate']).astype(int)
    
    # Añadimos ligera variación diaria al AOV para no hacerlo estático
    daily_aov = np.random.normal(aov, 15, len(df))
    df['revenue'] = round(df['sales'] * daily_aov, 2)
    
    return df

def main():
    """Pipeline de ejecución de la Fase 1."""
    print("Iniciando generación de datos sintéticos...")
    start_date = date(2022, 1, 1) # Usamos datos históricos para predecir el futuro
    
    df_base = generate_base_metrics(start_date, days=730)
    df_anomalies = inject_anomalies(df_base)
    df_final = calculate_revenue(df_anomalies)
    
    output_path = 'synthetic_business_data.csv'
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ Datos generados exitosamente en: {output_path}")
    print(f"Total de registros: {len(df_final)}")
    print(f"Anomalías inyectadas: {df_final['is_anomaly'].sum()}")
    print("\nMuestra de datos:")
    print(df_final.head())

if __name__ == "__main__":
    main()