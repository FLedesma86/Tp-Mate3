import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_DATOS = os.path.join(CARPETA_PROYECTO, "liga_profesional_argentina_2023_2025_maestro.csv")
CARPETA_RESULTADOS = os.path.join(CARPETA_PROYECTO, "resultados")

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

if not os.path.exists(ARCHIVO_DATOS):
    raise FileNotFoundError(
        f"No se encontró el archivo: {ARCHIVO_DATOS}. "
        "Ejecutá primero el script base_maestra.py para consolidar los datos."
    )

df = pd.read_csv(ARCHIVO_DATOS)

features_for_correlation = ["HTSFT", "HSONFT", "HBPFT", "HCFT"]
target_col = "FTHG"

for col in features_for_correlation + [target_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df_modelo = df[features_for_correlation + [target_col]].dropna()

X = df_modelo[features_for_correlation]
y = df_modelo[target_col]


print("1. SUPUESTOS PRE-MODELO")

plt.figure(figsize=(12, 10))
for i, feature in enumerate(features_for_correlation, 1):
    plt.subplot(2, 2, i)
    plt.scatter(df_modelo[feature], y, alpha=0.5)
    plt.title(f"{feature} vs. {target_col}")
    plt.xlabel(feature)
    plt.ylabel(target_col)
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "pre_linealidad_scatter.png"), dpi=160)
plt.close()

X_vif_example = df_modelo[features_for_correlation].copy()
X_vif_example = sm.add_constant(X_vif_example)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif_example.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif_example.values, i)
    for i in range(X_vif_example.shape[1])
]
print("\nFactor de Inflación de la Varianza (VIF):\n", vif_data)
print("\nUn VIF > 5 o > 10 generalmente indica multicolinealidad significativa.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler_X_multi = StandardScaler()
X_train_scaled_multi = scaler_X_multi.fit_transform(X_train)
X_test_scaled_multi = scaler_X_multi.transform(X_test)

regression_multi = LinearRegression()
regression_multi.fit(X_train_scaled_multi, y_train)

print("\nModelo de Regresión Lineal Múltiple entrenado.")

print("2. EVALUACIÓN Y SUPUESTOS POST-MODELO")

y_pred_multi = regression_multi.predict(X_test_scaled_multi)
residuals_multi = y_test - y_pred_multi

print("\nPrimeras 5 predicciones:\n", y_pred_multi[:5])
print("\nPrimeros 5 residuos:\n", residuals_multi[:5])

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_multi, residuals_multi, alpha=0.7)
plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
plt.title("Gráfico de Residuos vs. Valores Predichos (RLM)")
plt.xlabel("Valores Predichos")
plt.ylabel("Residuos")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "post_residuos_vs_predichos.png"), dpi=160)
plt.close()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(residuals_multi, bins=20, kde=True)
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuos con KDE (RLM)")
plt.axvline(0, color="r", linestyle="--", label="Cero")
plt.legend()

plt.subplot(1, 2, 2)
sm.qqplot(residuals_multi, line="s", ax=plt.gca())
plt.title("Gráfico Q-Q de Residuos (RLM)")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "post_normalidad_residuos.png"), dpi=160)
plt.close()

shapiro_test_multi = shapiro(residuals_multi)
print(f"\nEstadístico de Shapiro-Wilk (RLM): {shapiro_test_multi[0]:.4f}")
print(f"Valor p (RLM): {shapiro_test_multi[1]:.4f}")
if shapiro_test_multi[1] > 0.05:
    print("Los residuos parecen seguir una distribución normal (no se rechaza H0).")
else:
    print("Los residuos no parecen seguir una distribución normal (se rechaza H0).")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals_multi)), residuals_multi, alpha=0.7)
plt.xlabel("Orden de Recopilación de Datos (Índice)")
plt.ylabel("Residuos")
plt.title("Gráfico de Residuos vs. Orden de Recopilación de Datos (RLM)")
plt.axhline(y=0, color="r", linestyle="--", linewidth=2)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "post_residuos_orden.png"), dpi=160)
plt.close()

dw_test_multi = durbin_watson(residuals_multi)
print(f"\nEstadístico de Durbin-Watson (RLM): {dw_test_multi:.4f}")
if dw_test_multi < 1.5:
    print("Posible autocorrelación positiva.")
elif dw_test_multi > 2.5:
    print("Posible autocorrelación negativa.")
else:
    print("Los errores parecen ser independientes (no hay evidencia de autocorrelación).")

plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred_multi, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Línea de ajuste perfecto",
)
plt.xlabel("Valores Reales (y_test)")
plt.ylabel("Valores Predichos (y_pred_multi)")
plt.title("Valores Reales vs. Predicciones (RLM)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "post_reales_vs_predicciones.png"), dpi=160)
plt.close()

print("3. MÉTRICAS DE EVALUACIÓN")

r_squared = r2_score(y_test, y_pred_multi)
n = X_test.shape[0]
p = X_test.shape[1]
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

mse = mean_squared_error(y_test, y_pred_multi)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_multi)

print(f"R-cuadrado (R2): {r_squared:.4f}")
print(f"R-cuadrado Ajustado: {adjusted_r_squared:.4f}")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")

print("4. PREDICCIÓN CON NUEVO DATO")

new_match_data = pd.DataFrame([
    {
        "HTSFT": 12.0,
        "HSONFT": 5.0,
        "HBPFT": 55.0,
        "HCFT": 6.0
    }
])[features_for_correlation]

print("Nuevo dato de partido (sin escalar):")
print(new_match_data)

new_match_scaled = scaler_X_multi.transform(new_match_data)

predicted_goals = regression_multi.predict(new_match_scaled)
print(f"\nLos goles predichos para el equipo local en el nuevo partido son: {predicted_goals[0]:.2f}")