import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
ARCHIVO = os.path.join(CARPETA_PROYECTO, "liga_profesional_argentina_2023_2025_maestro.csv")
CARPETA_RESULTADOS = os.path.join(CARPETA_PROYECTO, "resultados")

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

if not os.path.exists(ARCHIVO):
    raise FileNotFoundError(
        f"No se encontró la base maestra: {ARCHIVO}. "
        "Primero ejecutá armar_base_maestra.py."
    )

df = pd.read_csv(ARCHIVO)

columnas_numericas = [
    "FTHG", "FTAG", "HTSFT", "ATSFT", "HSONFT", "ASONFT",
    "HBPFT", "ABPFT", "HCFT", "ACFT"
]
for columna in columnas_numericas:
    if columna in df.columns:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

df["TotalGoles"] = df["FTHG"] + df["FTAG"]
df["RematesTotales"] = df["HTSFT"] + df["ATSFT"]
df["RematesAlArco"] = df["HSONFT"] + df["ASONFT"]
df["PosesionPromedio"] = (df["HBPFT"] + df["ABPFT"]) / 2
df["CornersTotales"] = df["HCFT"] + df["ACFT"]

print(" CONTROL DE CALIDAD ")
print(f"Partidos: {len(df)}")
print(f"IDs duplicados: {df['id'].duplicated().sum()}")
print(f"Valores faltantes en la base original: {pd.read_csv(ARCHIVO).isna().sum().sum()}")

print(" ESTADÍSTICAS GENERALES ")
print(f"Goles totales: {int(df['TotalGoles'].sum())}")
print(f"Promedio de goles por partido: {df['TotalGoles'].mean():.3f}")
print(f"Promedio de goles del local: {df['FTHG'].mean():.3f}")
print(f"Promedio de goles del visitante: {df['FTAG'].mean():.3f}")

resumen = df.groupby("Season").agg(
    Partidos=("id", "count"),
    Goles=("TotalGoles", "sum"),
    PromedioGoles=("TotalGoles", "mean"),
    PromedioLocal=("FTHG", "mean"),
    PromedioVisitante=("FTAG", "mean"),
    MedianaGoles=("TotalGoles", "median"),
    DesvGoles=("TotalGoles", "std")
).round(3)

print(" RESUMEN POR TEMPORADA ")
print(resumen.to_string())

variables = [
    "TotalGoles", "RematesTotales", "RematesAlArco",
    "PosesionPromedio", "CornersTotales"
]
correlaciones = df[variables].corr()["TotalGoles"].drop("TotalGoles").sort_values(
    ascending=False
).round(3)

print(" CORRELACIÓN DE PEARSON CON GOLES TOTALES ")
print(correlaciones.to_string())

resumen.to_csv(os.path.join(CARPETA_RESULTADOS, "resumen_por_temporada.csv"),
               encoding="utf-8-sig")
correlaciones.rename("Correlacion_con_goles").to_csv(
    os.path.join(CARPETA_RESULTADOS, "correlaciones_con_goles.csv"),
    encoding="utf-8-sig"
)

# Gráfico 1: distribución de goles por partido.
distribucion = df["TotalGoles"].value_counts().sort_index()
plt.figure()
distribucion.plot(kind="bar")
plt.title("Distribución de goles por partido (2023–2025)")
plt.xlabel("Goles totales en el partido")
plt.ylabel("Cantidad de partidos")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "distribucion_goles.png"), dpi=160)
plt.close()

# Gráfico 2: promedio de goles por año.
plt.figure()
resumen["PromedioGoles"].plot(kind="bar")
plt.title("Promedio de goles por partido según temporada")
plt.xlabel("Temporada")
plt.ylabel("Goles por partido")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "promedio_goles_temporada.png"), dpi=160)
plt.close()

# Gráfico 3: correlaciones con goles totales.
plt.figure()
correlaciones.plot(kind="bar")
plt.axhline(0, linewidth=0.8)
plt.title("Correlación de Pearson con goles totales")
plt.xlabel("Variable")
plt.ylabel("Correlación")
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_RESULTADOS, "correlaciones_goles.png"), dpi=160)
plt.close()

print(f"\nGráficos y tablas guardados en: {CARPETA_RESULTADOS}")
