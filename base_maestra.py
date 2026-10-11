from pathlib import Path
import pandas as pd

CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA_PROYECTO / "datos"
ARCHIVO_SALIDA = CARPETA_PROYECTO / "liga_profesional_argentina_2023_2025_maestro.csv"
ARCHIVO_AUDITORIA = CARPETA_PROYECTO / "auditoria_base_liga_2023_2025.txt"

ARCHIVOS = {
    "overview": CARPETA_DATOS / "Database - Overview - Argentina Liga Profesional 23-24-25 - LS.csv",
    "scores": CARPETA_DATOS / "Database - Scores - Argentina Liga Profesional 23-24-25 LS.csv",
    "attack": CARPETA_DATOS / "Database - Attack  Poss - Argentina Liga Profesional 23-24-25 - LS.csv",
    "corners": CARPETA_DATOS / "Database - Corners  Cards - Argentina Liga Profesional 23-24-25 - LS.csv",
}

faltantes = [str(ruta) for ruta in ARCHIVOS.values() if not ruta.exists()]
if faltantes:
    mensaje = "No se encontraron estos CSV dentro de la carpeta 'datos':\n- " + "\n- ".join(faltantes)
    raise FileNotFoundError(mensaje)

datos = {nombre: pd.read_csv(ruta) for nombre, ruta in ARCHIVOS.items()}
ids_base = set(datos["overview"]["id"])

for nombre, df in datos.items():
    if "id" not in df.columns:
        raise ValueError(f"{nombre}: no existe la columna 'id'.")
    if df["id"].duplicated().any():
        raise ValueError(f"{nombre}: hay IDs duplicados.")
    if set(df["id"]) != ids_base:
        raise ValueError(f"{nombre}: los IDs no coinciden con Overview.")

columnas_comunes = ["matchDate", "Country", "League", "Season", "homeTeam", "awayTeam"]
referencia = datos["overview"].set_index("id").sort_index()
for nombre in ["scores", "attack", "corners"]:
    otro = datos[nombre].set_index("id").sort_index()
    for columna in columnas_comunes:
        if columna in otro.columns and columna in referencia.columns:
            if not referencia[columna].equals(otro[columna]):
                raise ValueError(f"Diferencias en {columna} entre Overview y {nombre}.")

maestro = datos["overview"].copy()
for nombre in ["scores", "attack", "corners"]:
    columnas = ["id"] + [c for c in datos[nombre].columns if c not in maestro.columns]
    maestro = maestro.merge(
        datos[nombre][columnas], on="id", how="inner", validate="one_to_one"
    )

maestro = maestro.sort_values(["Season", "matchDate", "id"]).reset_index(drop=True)
maestro.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")

lineas = [
    "AUDITORÍA DE LA BASE MAESTRA — LIGA PROFESIONAL ARGENTINA 2023–2025",
    f"Archivo generado: {ARCHIVO_SALIDA.name}",
    f"Filas: {len(maestro)}",
    f"Columnas: {len(maestro.columns)}",
    f"IDs duplicados en la base final: {maestro['id'].duplicated().sum()}",
    f"Valores faltantes en la base final: {maestro.isna().sum().sum()}",
    "",
    "Archivos de origen:",
]
for nombre, df in datos.items():
    lineas.append(
        f"- {ARCHIVOS[nombre].name}: {len(df)} filas; "
        f"IDs únicos={df['id'].nunique()}; duplicados={df['id'].duplicated().sum()}; "
        f"valores faltantes={df.isna().sum().sum()}"
    )
lineas.extend(["", "Partidos por temporada:", maestro["Season"].value_counts().sort_index().to_string()])
ARCHIVO_AUDITORIA.write_text("\n".join(lineas) + "\n", encoding="utf-8")

print("Base creada correctamente:", ARCHIVO_SALIDA)
print("Filas:", len(maestro))
print("Columnas:", len(maestro.columns))
print("Partidos por temporada:")
print(maestro["Season"].value_counts().sort_index())
print("Auditoría guardada en:", ARCHIVO_AUDITORIA)
