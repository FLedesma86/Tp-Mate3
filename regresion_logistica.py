"""
Regresión logística y regularizaciones — Liga Profesional Argentina (2023–2025)

Pregunta: ¿qué características del partido (remates y córners) se asocian con que
termine con muchos goles (3 o más)?

Usa las variables derivadas definidas en el README:
  TotalGoles, RematesTotales, RematesAlArco, PosesionPromedio, CornersTotales.
Como son variables del partido completo (suman local + visitante), el objetivo binario
se define a partir de TotalGoles. TotalGoles no se usa como predictor porque define el objetivo.

Sigue el esquema visto en clase: verificación de supuestos (multicolinealidad con VIF),
escalado, entrenamiento, evaluación (accuracy, matriz de confusión, precision/recall/F1,
ROC/AUC), validación cruzada y predicción de un caso nuevo. Sobre eso se agregan las
regularizaciones L1, L2 y ElasticNet y la elección del hiperparámetro C.

Requisitos: pandas, numpy, matplotlib, scikit-learn y statsmodels
(pip install scikit-learn statsmodels).
Ejecutar después de base_maestra.py (usa la base maestra ya generada).
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, log_loss, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore")

# Un partido se considera "de muchos goles" si tiene al menos este total (3 = más de 2,5 goles).
UMBRAL_GOLES = 3

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
ARCHIVO = os.path.join(CARPETA_PROYECTO, "liga_profesional_argentina_2023_2025_maestro.csv")
CARPETA_RESULTADOS = os.path.join(CARPETA_PROYECTO, "resultados/regresion_logistica")
SEMILLA = 42

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
if not os.path.exists(ARCHIVO):
    raise FileNotFoundError(
        f"No se encontró la base maestra: {ARCHIVO}. Primero ejecutá base_maestra.py."
    )

# ---------------------------------------------------------------------------
# 1. Datos, variables derivadas y variable objetivo
# ---------------------------------------------------------------------------
df = pd.read_csv(ARCHIVO)

columnas_numericas = ["FTHG", "FTAG", "HTSFT", "ATSFT", "HSONFT", "ASONFT", "HBPFT", "ABPFT", "HCFT", "ACFT"]
for columna in columnas_numericas:
    df[columna] = pd.to_numeric(df[columna], errors="coerce")

# Variables derivadas (las mismas del README y de analisis_exploratorio.py)
df["TotalGoles"] = df["FTHG"] + df["FTAG"]
df["RematesTotales"] = df["HTSFT"] + df["ATSFT"]
df["RematesAlArco"] = df["HSONFT"] + df["ASONFT"]
df["PosesionPromedio"] = (df["HBPFT"] + df["ABPFT"]) / 2
df["CornersTotales"] = df["HCFT"] + df["ACFT"]

# Objetivo binario: 1 si el partido tuvo UMBRAL_GOLES goles o más; 0 si tuvo menos.
df["MuchosGoles"] = (df["TotalGoles"] >= UMBRAL_GOLES).astype(int)

# Predictores: remates totales, remates al arco y córners totales.
# PosesionPromedio NO se usa: la posesión del local y la del visitante suman 100, así que su promedio
# es siempre 50 (no varía) y no aporta información; estandarizada, amplificaría ruido.
VARIABLES = ["RematesTotales", "RematesAlArco", "CornersTotales"]
df = df.dropna(subset=VARIABLES + ["MuchosGoles"])

# Partición temporal: se entrena con 2023–2024 y se prueba con 2025 (temporada no vista).
entrenamiento = df[df["Season"] <= 2024]
prueba = df[df["Season"] == 2025]
X_tr, y_tr = entrenamiento[VARIABLES], entrenamiento["MuchosGoles"]
X_te, y_te = prueba[VARIABLES], prueba["MuchosGoles"]

print(" REGRESIÓN LOGÍSTICA ")
print(f"Objetivo: partido con {UMBRAL_GOLES} o más goles")
print(f"Entrenamiento (2023–2024): {len(X_tr)} partidos | Prueba (2025): {len(X_te)} partidos")
print(f"Proporción de partidos con muchos goles -> entrenamiento: {y_tr.mean():.3f} | prueba: {y_te.mean():.3f}")
print(f"\nControl de PosesionPromedio: desvío estándar = {df['PosesionPromedio'].std():.4f} "
      f"(valores distintos de 50: {(df['PosesionPromedio'] != 50).sum()} de {len(df)}) -> se excluye del modelo")

# ---------------------------------------------------------------------------
# 2. Verificación de supuestos
# ---------------------------------------------------------------------------
# a) Variable binaria: MuchosGoles toma solo los valores 0 y 1.
# b) Tamaño de muestra: regla de 10–20 eventos por variable (se usa la clase menos frecuente).
eventos = int(min(y_tr.sum(), (1 - y_tr).sum()))
print("\n VERIFICACIÓN DE SUPUESTOS ")
print(f"Eventos de la clase menos frecuente: {eventos} | por variable: {eventos / len(VARIABLES):.1f} (mínimo recomendado: 10–20)")

# c) Ausencia de multicolinealidad: VIF (> 5 o > 10 indica correlación fuerte con otras variables).
X_const = add_constant(X_tr)
vif = pd.DataFrame({
    "Variable": X_const.columns,
    "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
})
vif = vif[vif["Variable"] != "const"].sort_values("VIF", ascending=False).reset_index(drop=True)
print("Factor de inflación de la varianza (VIF):")
print(vif.round(2).to_string(index=False))
if (vif["VIF"] > 10).any():
    print("Atención: hay variables con VIF > 10; conviene eliminarlas o regularizar.")
else:
    print("Ninguna variable supera VIF = 10: no hay multicolinealidad problemática.")

# ---------------------------------------------------------------------------
# 3. Modelos con distintas regularizaciones
#    C es la INVERSA de la fuerza de regularización: C chico = más regularización.
#    Se elige por validación cruzada (5 particiones estratificadas) minimizando la pérdida logarítmica.
#    Las variables se estandarizan (StandardScaler) dentro de un Pipeline, así el escalado se ajusta
#    solo con los datos de entrenamiento de cada partición.
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
GRILLA_C = np.logspace(-3, 2, 20)


def armar(clasificador):
    return Pipeline([("escalado", StandardScaler()), ("modelo", clasificador)])


def buscar(clasificador, grilla):
    gs = GridSearchCV(armar(clasificador), {f"modelo__{k}": v for k, v in grilla.items()},
                      cv=cv, scoring="neg_log_loss", n_jobs=-1)
    return gs.fit(X_tr, y_tr)


modelos = {
    "Sin regularización": armar(LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)).fit(X_tr, y_tr),
    "L2 (Ridge)": buscar(LogisticRegression(penalty="l2", solver="lbfgs", max_iter=10000), {"C": GRILLA_C}),
    "L1 (Lasso)": buscar(LogisticRegression(penalty="l1", solver="liblinear", max_iter=10000), {"C": GRILLA_C}),
    "ElasticNet": buscar(LogisticRegression(penalty="elasticnet", solver="saga", max_iter=20000),
                         {"C": GRILLA_C, "l1_ratio": [0.25, 0.5, 0.75]}),
}


def estimador(modelo):
    return modelo.best_estimator_ if isinstance(modelo, GridSearchCV) else modelo


# Modelo elegido: el regularizado con mejor pérdida logarítmica en validación cruzada
# (se decide con datos de entrenamiento, sin mirar la temporada de prueba).
mejor = max((n for n, m in modelos.items() if isinstance(m, GridSearchCV)),
            key=lambda n: modelos[n].best_score_)
modelo_final = estimador(modelos[mejor])
print(f"\nModelo elegido por validación cruzada: {mejor}")

# ---------------------------------------------------------------------------
# 4. Evaluación sobre la temporada de prueba
# ---------------------------------------------------------------------------
def metricas(y, proba, umbral=0.5):
    pred = (proba >= umbral).astype(int)
    return {"Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, proba),
            "LogLoss": log_loss(y, proba)}


filas = []
base = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
filas.append({"Modelo": "Base (proporción de partidos con muchos goles)", "Hiperparámetros": "-",
              **metricas(y_te, base.predict_proba(X_te)[:, 1])})

probas = {}
for nombre, modelo in modelos.items():
    probas[nombre] = estimador(modelo).predict_proba(X_te)[:, 1]
    hiper = ({k.replace("modelo__", ""): (float(round(v, 4)) if not isinstance(v, str) else v)
              for k, v in modelo.best_params_.items()} if isinstance(modelo, GridSearchCV) else "-")
    filas.append({"Modelo": nombre, "Hiperparámetros": hiper, **metricas(y_te, probas[nombre])})

tabla = pd.DataFrame(filas)
print("\n MÉTRICAS EN LA TEMPORADA 2025 (umbral 0,5) ")
print(tabla.round(3).to_string(index=False))

# Matriz de confusión y reporte de clasificación del modelo elegido
ETIQUETAS = [f"Menos de {UMBRAL_GOLES} goles (0)", f"{UMBRAL_GOLES} o más goles (1)"]
pred_final = (probas[mejor] >= 0.5).astype(int)
matriz = confusion_matrix(y_te, pred_final)
reporte = classification_report(y_te, pred_final, target_names=ETIQUETAS, zero_division=0)
print(f"\n MATRIZ DE CONFUSIÓN — {mejor} ")
print(pd.DataFrame(matriz, index=["Actual 0", "Actual 1"], columns=["Predicho 0", "Predicho 1"]).to_string())
print(f"\n REPORTE DE CLASIFICACIÓN — {mejor} ")
print(reporte)

# ---------------------------------------------------------------------------
# 4b. Desbalance de clases: efecto del umbral y de class_weight="balanced"
#     Solo ~1 de cada 3 partidos tiene muchos goles, así que con el umbral 0,5 el modelo casi siempre
#     predice la clase mayoritaria (recall bajo). Hay dos formas de corregirlo:
#       (a) bajar el umbral de decisión;
#       (b) ponderar las clases (class_weight="balanced"), como se vio en clase.
#     Ambas suben el recall a costa de la precisión. La tabla de umbrales es descriptiva:
#     el umbral no se eligió mirando la temporada de prueba.
# ---------------------------------------------------------------------------
umbrales = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
tabla_umbral = pd.DataFrame([
    {"Umbral": u, **{k: v for k, v in metricas(y_te, probas[mejor], u).items()
                     if k in ["Accuracy", "Precision", "Recall", "F1"]}}
    for u in umbrales
])
print(f"\n EFECTO DEL UMBRAL DE DECISIÓN — {mejor} (temporada 2025) ")
print(tabla_umbral.round(3).to_string(index=False))

balanceado = GridSearchCV(
    armar(LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=10000)),
    {"modelo__C": GRILLA_C}, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
proba_bal = balanceado.predict_proba(X_te)[:, 1]
tabla_bal = pd.DataFrame([{
    "Modelo": "L2 con class_weight='balanced'",
    "Hiperparámetros": {"C": float(round(balanceado.best_params_["modelo__C"], 4))},
    **metricas(y_te, proba_bal)}])
print("\n L2 CON CLASES BALANCEADAS (umbral 0,5) ")
print(tabla_bal.round(3).to_string(index=False))
print("Nota: al ponderar las clases las probabilidades dejan de estar calibradas (la pérdida logarítmica empeora),"
      " pero el recall mejora.")

# ---------------------------------------------------------------------------
# 5. Coeficientes y odds ratios (variables estandarizadas: se pueden comparar entre sí)
#    Odds ratio = exp(coeficiente): cuánto se multiplican los odds de que el partido tenga
#    muchos goles cuando la variable aumenta 1 desvío estándar.
# ---------------------------------------------------------------------------
coeficientes = pd.DataFrame(
    {nombre: estimador(modelo).named_steps["modelo"].coef_[0] for nombre, modelo in modelos.items()},
    index=VARIABLES,
)
odds = pd.DataFrame({"Coeficiente": coeficientes[mejor], "OddsRatio": np.exp(coeficientes[mejor])})
print("\n COEFICIENTES (variables estandarizadas) ")
print(coeficientes.round(3).to_string())
print(f"\n ODDS RATIOS — {mejor} (por cada +1 desvío estándar de la variable) ")
print(odds.round(3).to_string())

nulos_l1 = coeficientes.index[coeficientes["L1 (Lasso)"] == 0].tolist()
print("\nVariables que L1 deja en cero:", nulos_l1 if nulos_l1 else "ninguna")

# ---------------------------------------------------------------------------
# 6. Validación cruzada del modelo elegido (sobre el conjunto de entrenamiento)
#    La media y el desvío entre particiones muestran qué tan estable es el rendimiento.
#    Nota: C se eligió con estos mismos datos, por lo que la estimación es algo optimista.
# ---------------------------------------------------------------------------
resultados_cv = {}
for nombre_metrica, scoring in [("Accuracy", "accuracy"), ("Precision", "precision"),
                                ("Recall", "recall"), ("F1", "f1"), ("ROC AUC", "roc_auc")]:
    puntajes = cross_val_score(clone(modelo_final), X_tr, y_tr, cv=cv, scoring=scoring)
    resultados_cv[nombre_metrica] = (puntajes.mean(), puntajes.std())
print(f"\n VALIDACIÓN CRUZADA (K=5) — {mejor} ")
for nombre_metrica, (media, desvio) in resultados_cv.items():
    print(f"{nombre_metrica}: {media:.4f} (+/- {desvio:.4f})")

# ---------------------------------------------------------------------------
# 7. Predicción de un partido nuevo
#    Ejemplo: un partido "promedio" (media de cada variable en el entrenamiento).
# ---------------------------------------------------------------------------
partido_nuevo = pd.DataFrame([X_tr.mean()])
proba_nuevo = modelo_final.predict_proba(partido_nuevo)[0]
clase_nueva = int(proba_nuevo[1] >= 0.5)
print("\n PREDICCIÓN DE UN PARTIDO PROMEDIO ")
print(partido_nuevo.round(2).to_string(index=False))
print(f"Probabilidad de menos de {UMBRAL_GOLES} goles (0): {proba_nuevo[0]:.4f}")
print(f"Probabilidad de {UMBRAL_GOLES} o más goles (1): {proba_nuevo[1]:.4f}")
print("Clase predicha:", f"{UMBRAL_GOLES} o más goles" if clase_nueva == 1 else f"menos de {UMBRAL_GOLES} goles")

# ---------------------------------------------------------------------------
# 8. Archivos de salida
# ---------------------------------------------------------------------------
pd.concat([tabla, tabla_bal]).to_csv(os.path.join(CARPETA_RESULTADOS, "logistica_metricas.csv"), index=False, encoding="utf-8-sig")
tabla_umbral.to_csv(os.path.join(CARPETA_RESULTADOS, "logistica_umbral.csv"), index=False, encoding="utf-8-sig")
coeficientes.to_csv(os.path.join(CARPETA_RESULTADOS, "logistica_coeficientes.csv"), encoding="utf-8-sig")
vif.to_csv(os.path.join(CARPETA_RESULTADOS, "logistica_vif.csv"), index=False, encoding="utf-8-sig")
odds.to_csv(os.path.join(CARPETA_RESULTADOS, "logistica_odds_ratios.csv"), encoding="utf-8-sig")

# Gráfico 1: trayectoria de los coeficientes según la fuerza de regularización (L1 y L2).
escalador = StandardScaler().fit(X_tr)
X_tr_esc = escalador.transform(X_tr)
valores_c = np.logspace(-3, 2, 40)
fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for eje, (penalidad, solver, titulo) in zip(ejes, [("l2", "lbfgs", "L2 (Ridge)"), ("l1", "liblinear", "L1 (Lasso)")]):
    trayectoria = np.array([
        LogisticRegression(penalty=penalidad, C=c, solver=solver, max_iter=10000).fit(X_tr_esc, y_tr).coef_[0]
        for c in valores_c
    ])
    for j, variable in enumerate(VARIABLES):
        eje.plot(valores_c, trayectoria[:, j], label=variable)
    eje.set_xscale("log")
    eje.axhline(0, color="k", lw=0.8)
    eje.set_title(f"Trayectoria de coeficientes — {titulo}")
    eje.set_xlabel("C (menor C = más regularización)")
ejes[0].set_ylabel("Coeficiente (variables estandarizadas)")
ejes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "logistica_trayectoria_regularizacion.png"), dpi=160)
plt.close()

# Gráfico 2: curvas ROC de todos los modelos.
plt.figure(figsize=(6, 5))
for nombre, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_te, proba)
    plt.plot(fpr, tpr, label=f"{nombre} (AUC = {roc_auc_score(y_te, proba):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Azar")
plt.title("Curvas ROC — temporada 2025")
plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "logistica_roc.png"), dpi=160)
plt.close()

# Gráfico 3: matriz de confusión del modelo elegido.
ConfusionMatrixDisplay(matriz, display_labels=[f"< {UMBRAL_GOLES} goles", f"≥ {UMBRAL_GOLES} goles"]).plot(
    cmap="Blues", colorbar=False)
plt.title(f"Matriz de confusión — {mejor}")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "logistica_matriz_confusion.png"), dpi=160)
plt.close()

# Gráfico 4: frontera de decisión con dos variables (remates al arco y remates totales);
# la otra variable (córners) se mantiene en su valor promedio. Se muestra en entrenamiento y prueba.
var1, var2 = "RematesAlArco", "RematesTotales"
fig, ejes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
x_min, x_max = df[var1].min() - 1, df[var1].max() + 1
y_min, y_max = df[var2].min() - 1, df[var2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
malla = pd.DataFrame(np.tile(X_tr.mean().values, (xx.size, 1)), columns=VARIABLES)
malla[var1], malla[var2] = xx.ravel(), yy.ravel()
Z = modelo_final.predict_proba(malla)[:, 1].reshape(xx.shape)
for eje, (X_, y_, titulo) in zip(ejes, [(X_tr, y_tr, "Entrenamiento (2023–2024)"), (X_te, y_te, "Prueba (2025)")]):
    relleno = eje.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21), cmap="RdYlGn", alpha=0.85)
    eje.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles="--", linewidths=2)
    eje.scatter(X_[var1], X_[var2], c=np.where(y_ == 1, "#1b7837", "#b2182b"), s=18, edgecolor="k", linewidth=0.3, alpha=0.8)
    eje.set_title(f"Frontera de decisión — {titulo}")
    eje.set_xlabel("Remates al arco (local + visitante)"); eje.set_ylabel("Remates totales (local + visitante)")
fig.colorbar(relleno, ax=ejes, label=f"P({UMBRAL_GOLES} o más goles)")
plt.savefig(os.path.join(CARPETA_RESULTADOS, "logistica_frontera_decision.png"), dpi=160, bbox_inches="tight")
plt.close()

# Resumen en texto
with open(os.path.join(CARPETA_RESULTADOS, "logistica_resumen.txt"), "w", encoding="utf-8") as f:
    f.write("REGRESIÓN LOGÍSTICA Y REGULARIZACIONES — LIGA PROFESIONAL ARGENTINA 2023–2025\n")
    f.write("=" * 70 + "\n")
    f.write(f"Objetivo: partido con {UMBRAL_GOLES} o más goles\n")
    f.write(f"Entrenamiento (2023–2024): {len(X_tr)} partidos | Prueba (2025): {len(X_te)} partidos\n")
    f.write(f"Variables: {', '.join(VARIABLES)}\n")
    f.write("PosesionPromedio excluida: vale 50 en casi todos los partidos (local + visitante = 100).\n")
    f.write(f"Eventos por variable (clase menos frecuente): {eventos / len(VARIABLES):.1f}\n\n")
    f.write("VIF\n" + vif.round(2).to_string(index=False) + "\n\n")
    f.write("MÉTRICAS EN LA TEMPORADA 2025 (umbral 0,5)\n" + tabla.round(3).to_string(index=False) + "\n\n")
    f.write(f"Modelo elegido por validación cruzada: {mejor}\n\n")
    f.write(f"EFECTO DEL UMBRAL — {mejor}\n" + tabla_umbral.round(3).to_string(index=False) + "\n\n")
    f.write("CLASES BALANCEADAS (umbral 0,5)\n" + tabla_bal.round(3).to_string(index=False) + "\n\n")
    f.write("REPORTE DE CLASIFICACIÓN\n" + reporte + "\n")
    f.write("COEFICIENTES (variables estandarizadas)\n" + coeficientes.round(3).to_string() + "\n\n")
    f.write("ODDS RATIOS (por +1 desvío estándar)\n" + odds.round(3).to_string() + "\n\n")
    f.write(f"Variables que L1 deja en cero: {nulos_l1 if nulos_l1 else 'ninguna'}\n\n")
    f.write("VALIDACIÓN CRUZADA (K=5, entrenamiento)\n")
    for nombre_metrica, (media, desvio) in resultados_cv.items():
        f.write(f"{nombre_metrica}: {media:.4f} (+/- {desvio:.4f})\n")
    f.write(f"\nPartido promedio -> P({UMBRAL_GOLES} o más goles) = {proba_nuevo[1]:.4f}\n")
    f.write("\nNota: remates y córners se conocen una vez jugado el partido; el modelo describe qué se asocia "
            "con los partidos de muchos goles, no los predice antes de jugarse.\n")

print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")
