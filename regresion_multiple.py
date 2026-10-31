import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    RidgeCV,
    LassoCV,
    SGDRegressor,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_DATOS = os.path.join(
    CARPETA_PROYECTO, "liga_profesional_argentina_2023_2025_maestro.csv"
)
CARPETA_RESULTADOS = os.path.join(CARPETA_PROYECTO, "resultados", "regresion_multiple")
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

correlation_matrix = df_modelo[features_for_correlation].corr()
plt.figure(figsize=(7, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Matriz de Correlación entre Variables Predictoras (RLM)")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "pre_matriz_correlacion_x.png"), dpi=160)
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
vif_sin_const = vif_data[vif_data["feature"] != "const"].reset_index(drop=True)
vif_sin_const.to_csv(
    os.path.join(CARPETA_RESULTADOS, "rlm_vif.csv"), index=False, encoding="utf-8-sig"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
scaler_X_multi = StandardScaler()
X_train_scaled_multi = scaler_X_multi.fit_transform(X_train)
X_test_scaled_multi = scaler_X_multi.transform(X_test)
regression_multi = LinearRegression()
regression_multi.fit(X_train_scaled_multi, y_train)
print("\nModelo de Regresión Lineal Múltiple entrenado.")

X_train_sm = sm.add_constant(
    pd.DataFrame(X_train_scaled_multi, columns=features_for_correlation, index=X_train.index)
)
modelo_sm = sm.OLS(y_train, X_train_sm).fit()
tabla_inferencia = pd.DataFrame({
    "Variable": modelo_sm.params.index,
    "Coeficiente": modelo_sm.params.values,
    "Error_Estandar": modelo_sm.bse.values,
    "t_stat": modelo_sm.tvalues.values,
    "p_valor": modelo_sm.pvalues.values,
    "IC_95_Inf": modelo_sm.conf_int()[0].values,
    "IC_95_Sup": modelo_sm.conf_int()[1].values,
})
tabla_inferencia.to_csv(
    os.path.join(CARPETA_RESULTADOS, "rlm_inferencia_coeficientes.csv"),
    index=False,
    encoding="utf-8-sig",
)
print(f"\nEstadístico F global: {modelo_sm.fvalue:.4f} (p-valor F: {modelo_sm.f_pvalue:.4e})")
print(tabla_inferencia.round(4).to_string(index=False))

print("2. EVALUACIÓN Y SUPUESTOS POST-MODELO")
y_train_pred = regression_multi.predict(X_train_scaled_multi)
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

def calcular_metricas_regresion(y_real, y_pred, n_features):
    n_obs = len(y_real)
    mse_val = mean_squared_error(y_real, y_pred)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(y_real, y_pred)
    r2_val = r2_score(y_real, y_pred)
    r2_adj_val = 1 - (1 - r2_val) * (n_obs - 1) / (n_obs - n_features - 1)
    media_y = np.mean(y_real)
    rmse_pct_media = (rmse_val / media_y) * 100
    return {
        "MAE": mae_val,
        "MSE": mse_val,
        "RMSE": rmse_val,
        "RMSE_%_Media": rmse_pct_media,
        "R2": r2_val,
        "R2_Ajustado": r2_adj_val,
    }

metricas_train = calcular_metricas_regresion(y_train, y_train_pred, p)
metricas_test = calcular_metricas_regresion(y_test, y_pred_multi, p)
df_diagnostico = pd.DataFrame([
    {"Conjunto": "Entrenamiento (Train 80%)", **metricas_train},
    {"Conjunto": "Validación / Prueba (Test 20%)", **metricas_test},
])
print("\nDiagnóstico Train vs. Test:")
print(df_diagnostico.round(4).to_string(index=False))

print("\n5. OPTIMIZACIÓN: DESCENSO DE GRADIENTE")
y_train_gd = y_train.values.reshape(-1, 1)
y_test_gd = y_test.values.reshape(-1, 1)

def gradient_descent(X_mat, y_vec, learning_rate, n_iterations):
    n_samples, n_features = X_mat.shape
    weights = np.zeros((n_features, 1))
    bias = 0.0
    cost_history = []
    for _ in range(n_iterations):
        y_predicted = np.dot(X_mat, weights) + bias
        error = y_predicted - y_vec
        cost = (1 / n_samples) * np.sum(error ** 2)
        cost_history.append(cost)
        d_weights = (2 / n_samples) * np.dot(X_mat.T, error)
        d_bias = (2 / n_samples) * np.sum(error)
        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias
    return weights, bias, cost_history

learning_rate_std = 0.01
n_iterations_std = 1000
weights_gd, bias_gd, cost_history_gd = gradient_descent(
    X_train_scaled_multi, y_train_gd, learning_rate_std, n_iterations_std
)
y_pred_gd = (np.dot(X_test_scaled_multi, weights_gd) + bias_gd).flatten()
mse_gd = mean_squared_error(y_test, y_pred_gd)
rmse_gd = np.sqrt(mse_gd)
r2_gd = r2_score(y_test, y_pred_gd)

sgd_model = SGDRegressor(
    max_iter=2000, learning_rate="constant", eta0=0.005, random_state=42, penalty=None
)
sgd_model.fit(X_train_scaled_multi, y_train)
y_pred_sgd = sgd_model.predict(X_test_scaled_multi)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
r2_sgd = r2_score(y_test, y_pred_sgd)

print(f"Coeficientes OLS:       {regression_multi.coef_.round(4)}")
print(f"Coeficientes GD Batch:  {weights_gd.flatten().round(4)}")
print(f"Coeficientes SGD:       {sgd_model.coef_.round(4)}")
print(f"Intercepto -> OLS: {regression_multi.intercept_:.4f} | GD Batch: {bias_gd:.4f} | SGD: {sgd_model.intercept_[0]:.4f}")
print(f"GD Batch Manual -> MSE: {mse_gd:.4f} | RMSE: {rmse_gd:.4f} | R2: {r2_gd:.4f}")
print(f"SGD Estocástico -> MSE: {mse_sgd:.4f} | RMSE: {np.sqrt(mse_sgd):.4f} | R2: {r2_sgd:.4f}")

tasas_lr = [0.001, 0.01, 0.1]
plt.figure(figsize=(9, 5))
for lr_val in tasas_lr:
    _, _, hist = gradient_descent(X_train_scaled_multi, y_train_gd, lr_val, 500)
    plt.plot(range(500), hist, lw=2, label=f"Learning Rate η = {lr_val}")
plt.title("Convergencia del Descenso de Gradiente según Tasa de Aprendizaje (η)")
plt.xlabel("Iteraciones (Epochs)")
plt.ylabel("Costo (MSE en Entrenamiento)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "opt_historial_costos_gd.png"), dpi=160)
plt.close()

print("\n6. REGULARIZACIÓN: RIDGE (L2) Y LASSO (L1)")
ridge_init = Ridge(alpha=0.5).fit(X_train_scaled_multi, y_train)
lasso_init = Lasso(alpha=0.1).fit(X_train_scaled_multi, y_train)
y_pred_ridge_init = ridge_init.predict(X_test_scaled_multi)
y_pred_lasso_init = lasso_init.predict(X_test_scaled_multi)
print(f"Ridge (alpha=0.5) -> MSE: {mean_squared_error(y_test, y_pred_ridge_init):.4f} | R2: {r2_score(y_test, y_pred_ridge_init):.4f}")
print(f"Lasso (alpha=0.1) -> MSE: {mean_squared_error(y_test, y_pred_lasso_init):.4f} | R2: {r2_score(y_test, y_pred_lasso_init):.4f}")

vars_cero_lasso_init = [
    feat for feat, coef in zip(features_for_correlation, lasso_init.coef_) if abs(coef) < 1e-4
]
print(f"Variables con coeficiente cercano a cero en Lasso (alpha=0.1): {vars_cero_lasso_init}")

alphas_log = np.logspace(-3, 2, 20)
ridge_cv_auto = RidgeCV(alphas=alphas_log, cv=5).fit(X_train_scaled_multi, y_train)
lasso_cv_auto = LassoCV(
    alphas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0], cv=5, random_state=42
).fit(X_train_scaled_multi, y_train)
print(f"RidgeCV mejor alpha: {ridge_cv_auto.alpha_:.4f} | R2 Test: {ridge_cv_auto.score(X_test_scaled_multi, y_test):.4f}")
print(f"LassoCV mejor alpha: {lasso_cv_auto.alpha_:.4f} | R2 Test: {lasso_cv_auto.score(X_test_scaled_multi, y_test):.4f}")

print("\n7. VALIDACIÓN CRUZADA (K-FOLD) Y GRIDSEARCHCV")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linear_model_cv = LinearRegression()
mse_scores_cv = -cross_val_score(
    linear_model_cv, X_train_scaled_multi, y_train, cv=kf, scoring="neg_mean_squared_error"
)
r2_scores_cv = cross_val_score(
    linear_model_cv, X_train_scaled_multi, y_train, cv=kf, scoring="r2"
)
print(f"MSE por fold:  {mse_scores_cv.round(4)}")
print(f"MSE promedio:  {np.mean(mse_scores_cv):.4f} (+/- {np.std(mse_scores_cv):.4f})")
print(f"R2 por fold:   {r2_scores_cv.round(4)}")
print(f"R2 promedio:   {np.mean(r2_scores_cv):.4f} (+/- {np.std(r2_scores_cv):.4f})")

param_grid_ridge = {"alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]}
grid_search_ridge = GridSearchCV(
    Ridge(), param_grid_ridge, cv=kf, scoring="neg_mean_squared_error"
)
grid_search_ridge.fit(X_train_scaled_multi, y_train)
best_ridge = grid_search_ridge.best_estimator_
y_pred_best_ridge = best_ridge.predict(X_test_scaled_multi)

param_grid_lasso = {"alpha": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 5.0]}
grid_search_lasso = GridSearchCV(
    Lasso(random_state=42), param_grid_lasso, cv=kf, scoring="neg_mean_squared_error"
)
grid_search_lasso.fit(X_train_scaled_multi, y_train)
best_lasso = grid_search_lasso.best_estimator_
y_pred_best_lasso = best_lasso.predict(X_test_scaled_multi)

print(
    f"Mejor Ridge -> alpha = {grid_search_ridge.best_params_['alpha']} | "
    f"MSE CV = {-grid_search_ridge.best_score_:.4f} | "
    f"MSE Test = {mean_squared_error(y_test, y_pred_best_ridge):.4f} | "
    f"R2 Test = {r2_score(y_test, y_pred_best_ridge):.4f}"
)
print(
    f"Mejor Lasso -> alpha = {grid_search_lasso.best_params_['alpha']} | "
    f"MSE CV = {-grid_search_lasso.best_score_:.4f} | "
    f"MSE Test = {mean_squared_error(y_test, y_pred_best_lasso):.4f} | "
    f"R2 Test = {r2_score(y_test, y_pred_best_lasso):.4f}"
)

df_coefs = pd.DataFrame(
    {
        "OLS (Sin Reg.)": regression_multi.coef_,
        f"Best Ridge (α={best_ridge.alpha})": best_ridge.coef_,
        f"Best Lasso (α={best_lasso.alpha})": best_lasso.coef_,
        "Lasso Fuerte (α=0.1)": lasso_init.coef_,
    },
    index=features_for_correlation,
)
df_coefs.to_csv(
    os.path.join(CARPETA_RESULTADOS, "rlm_comparacion_coeficientes.csv"),
    encoding="utf-8-sig",
)
print("\nComparación de Coeficientes:")
print(df_coefs.round(4).to_string())

plt.figure(figsize=(10, 5))
ind = np.arange(len(features_for_correlation))
width = 0.2
plt.bar(ind - 1.5 * width, regression_multi.coef_, width, label="OLS (Sin Reg.)", alpha=0.85)
plt.bar(ind - 0.5 * width, best_ridge.coef_, width, label=f"Best Ridge (α={best_ridge.alpha})", alpha=0.85)
plt.bar(ind + 0.5 * width, best_lasso.coef_, width, label=f"Best Lasso (α={best_lasso.alpha})", alpha=0.85)
plt.bar(ind + 1.5 * width, lasso_init.coef_, width, label="Lasso (α=0.1)", alpha=0.85)
plt.axhline(0, color="k", linewidth=0.8)
plt.xticks(ind, features_for_correlation)
plt.ylabel("Magnitud del Coeficiente")
plt.title("Comparación de Coeficientes: OLS vs. Ridge y Lasso")
plt.legend()
plt.grid(True, axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "reg_comparacion_coeficientes.png"), dpi=160)
plt.close()

alphas_path = np.logspace(-3, 1, 40)
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for eje, (ClaseModelo, titulo) in zip(ejes, [(Ridge, "Ridge (L2)"), (Lasso, "Lasso (L1)")]):
    trayectoria = np.array([
        ClaseModelo(alpha=a).fit(X_train_scaled_multi, y_train).coef_ for a in alphas_path
    ])
    for j, var_name in enumerate(features_for_correlation):
        eje.plot(alphas_path, trayectoria[:, j], lw=2, label=var_name)
    eje.set_xscale("log")
    eje.axhline(0, color="k", lw=0.8, linestyle="--")
    eje.set_title(f"Trayectoria de Coeficientes RLM — {titulo}")
    eje.set_xlabel("Fuerza de penalización α (escala log)")
    eje.grid(True, alpha=0.3)
ejes[0].set_ylabel("Valor del Coeficiente")
ejes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "reg_trayectoria_ridge_lasso.png"), dpi=160)
plt.close()

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
scaler_poly = StandardScaler()
X_train_poly_scaled = scaler_poly.fit_transform(X_train_poly)
X_test_poly_scaled = scaler_poly.transform(X_test_poly)
ridge_poly = RidgeCV(alphas=np.logspace(-2, 3, 30), cv=kf).fit(X_train_poly_scaled, y_train)
y_pred_poly = ridge_poly.predict(X_test_poly_scaled)
metricas_poly = calcular_metricas_regresion(y_test, y_pred_poly, X_test_poly.shape[1])

tabla_modelos = pd.DataFrame([
    {"Modelo": "RLM Base (OLS Analítico)", "Hiperparámetro": "-", **calcular_metricas_regresion(y_test, y_pred_multi, p)},
    {"Modelo": "RLM Descenso de Gradiente (Batch)", "Hiperparámetro": f"lr={learning_rate_std}, iter={n_iterations_std}", **calcular_metricas_regresion(y_test, y_pred_gd, p)},
    {"Modelo": "RLM SGDRegressor (Estocástico)", "Hiperparámetro": "eta0=0.005", **calcular_metricas_regresion(y_test, y_pred_sgd, p)},
    {"Modelo": "RLM Ridge Inicial (L2)", "Hiperparámetro": "alpha=0.5", **calcular_metricas_regresion(y_test, y_pred_ridge_init, p)},
    {"Modelo": "RLM Lasso Inicial (L1)", "Hiperparámetro": "alpha=0.1", **calcular_metricas_regresion(y_test, y_pred_lasso_init, p)},
    {"Modelo": "RLM Best Ridge (GridSearchCV)", "Hiperparámetro": f"alpha={best_ridge.alpha}", **calcular_metricas_regresion(y_test, y_pred_best_ridge, p)},
    {"Modelo": "RLM Best Lasso (GridSearchCV)", "Hiperparámetro": f"alpha={best_lasso.alpha}", **calcular_metricas_regresion(y_test, y_pred_best_lasso, p)},
    {"Modelo": "RLM Polinómica Grado 2 + Ridge", "Hiperparámetro": f"deg=2, alpha={ridge_poly.alpha_:.2f}", **metricas_poly},
])
tabla_modelos.to_csv(
    os.path.join(CARPETA_RESULTADOS, "rlm_metricas_modelos.csv"),
    index=False,
    encoding="utf-8-sig",
)
print("\nResumen Comparativo de Modelos:")
print(tabla_modelos.round(4).to_string(index=False))

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
pred_gd_manual = (np.dot(new_match_scaled, weights_gd) + bias_gd)[0, 0]
pred_best_ridge = best_ridge.predict(new_match_scaled)[0]
pred_best_lasso = best_lasso.predict(new_match_scaled)[0]
print(f"\nLos goles predichos para el equipo local en el nuevo partido son: {predicted_goals[0]:.2f}")
print(f"Predicción GD Manual: {pred_gd_manual:.4f} | Best Ridge: {pred_best_ridge:.4f} | Best Lasso: {pred_best_lasso:.4f}")

ruta_resumen = os.path.join(CARPETA_RESULTADOS, "rlm_resumen.txt")
with open(ruta_resumen, "w", encoding="utf-8") as f:
    f.write("REGRESIÓN LINEAL MÚLTIPLE: MÉTRICAS, VALIDACIÓN, REGULARIZACIÓN Y OPTIMIZACIÓN\n")
    f.write("=" * 80 + "\n\n")
    f.write("1. DIAGNÓSTICO TRAIN VS. TEST\n")
    f.write(df_diagnostico.round(4).to_string(index=False) + "\n\n")
    f.write("2. INFERENCIA ESTADÍSTICA (OLS)\n")
    f.write(f"Estadístico F: {modelo_sm.fvalue:.4f} (p-valor: {modelo_sm.f_pvalue:.4e})\n")
    f.write(tabla_inferencia.round(4).to_string(index=False) + "\n\n")
    f.write("3. VALIDACIÓN CRUZADA K-FOLD (K=5)\n")
    f.write(f"MSE promedio CV: {np.mean(mse_scores_cv):.4f} (+/- {np.std(mse_scores_cv):.4f})\n")
    f.write(f"R2 promedio CV:  {np.mean(r2_scores_cv):.4f} (+/- {np.std(r2_scores_cv):.4f})\n\n")
    f.write("4. COMPARACIÓN DE COEFICIENTES (OLS VS. RIDGE VS. LASSO)\n")
    f.write(df_coefs.round(4).to_string() + "\n\n")
    f.write("5. TABLA COMPARATIVA DE TODOS LOS MODELOS EN TEST\n")
    f.write(tabla_modelos.round(4).to_string(index=False) + "\n")