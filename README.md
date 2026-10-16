# Análisis exploratorio — Liga Profesional Argentina (2023–2025)

## Objetivo
Explorar los resultados y algunas estadísticas de los partidos de la Liga Profesional Argentina de las temporadas 2023, 2024 y 2025, como preparación para el análisis posterior y los modelos de regresión del TP de Matemática III.

## Fuente y alcance
- Fuente: archivos CSV históricos exportados desde Footiqo.
- Temporadas: 2023, 2024 y 2025.
- Base consolidada: `liga_profesional_argentina_2023_2025_maestro.csv`.
- Registros: 1.266 partidos.
- Variables originales: 53 columnas.
- Los CSV de Overview, Scores, Attack & Possession y Corners & Cards se combinaron por `id`.
- Los archivos de entrada compartían los mismos IDs únicos y las columnas comunes de identificación verificadas coincidían.

> La disponibilidad y cobertura de las estadísticas dependen de la exportación de origen. Conviene conservar los CSV originales para reproducir el proceso.

## Estructura sugerida del proyecto

```text
TP3/
├── Database - Overview - Argentina Liga Profesional 23-24-25 - LS.csv
├── Database - Scores - Argentina Liga Profesional 23-24-25 LS.csv
├── Database - Attack  Poss - Argentina Liga Profesional 23-24-25 - LS.csv
├── Database - Corners  Cards - Argentina Liga Profesional 23-24-25 - LS.csv
├── base_maestra.py
├── liga_profesional_argentina_2023_2025_maestro.csv
├── analisis_exploratorio.py
├── regresion_logistica.py
├── README.md
└── resultados/  (opcional, para gráficos y resúmenes)
    └── regresion_logistica/  (salidas del modelo logístico)
```

## Variables derivadas para el análisis
- `TotalGoles`: goles del local (`FTHG`) + goles del visitante (`FTAG`).
- `RematesTotales`: remates totales del local (`HTSFT`) + visitante (`ATSFT`).
- `RematesAlArco`: remates al arco del local (`HSONFT`) + visitante (`ASONFT`).
- `PosesionPromedio`: media de posesión del local (`HBPFT`) y visitante (`ABPFT`).
- `CornersTotales`: córners del local (`HCFT`) + visitante (`ACFT`).

Estas variables agregadas describen el partido como un todo. Para modelos por equipo, habrá que transformar cada encuentro en dos registros (local y visitante) o definir variables específicas de cada lado.

## Resultados exploratorios
Se generaron tres gráficos:
1. Distribución de goles por partido.
2. Promedio de goles por temporada.
3. Correlaciones de Pearson entre goles totales y estadísticas seleccionadas.

Los valores calculados a partir de la base se resumen en `resultados_exploratorio.txt`.

## Observaciones e interpretación
- El promedio de goles por partido permite comparar la frecuencia goleadora entre temporadas, pero debe interpretarse considerando que el formato y la cantidad de partidos pueden variar.
- Comparar el promedio de goles local y visitante permite describir diferencias observadas; no demuestra por sí solo una causa.
- La correlación de Pearson resume la asociación lineal entre dos variables. Una correlación baja no implica que no exista relación no lineal ni que la variable carezca de utilidad en un modelo multivariable.
- Las estadísticas de remates, posesión y córners corresponden al partido ya disputado; por lo tanto, su uso para explicar goles es análisis del rendimiento durante el encuentro, no una predicción previa al partido.
- Las correlaciones son descriptivas y no prueban causalidad.

---

## Regresión logística y regularizaciones

### Objetivo
Estudiar qué características del partido se asocian con que termine con **muchos goles** (3 o más, es decir, más de 2,5) usando regresión logística, y comparar distintas regularizaciones (L1, L2 y ElasticNet).

### Cómo ejecutarlo
1. Ejecutar `base_maestra.py` (genera la base maestra).
2. Ejecutar `regresion_logistica.py`.

Requiere `pandas`, `numpy`, `matplotlib`, `scikit-learn` y `statsmodels` (`pip install scikit-learn statsmodels`). Los resultados se guardan en `resultados/regresion_logistica/`.

### Variable objetivo y variables predictoras
- **Objetivo (`MuchosGoles`)**: 1 si `TotalGoles` ≥ 3 y 0 si es menor. Es el 31,8% de los partidos y se mantiene estable entre temporadas (33,1% en 2023, 29,9% en 2024 y 32,4% en 2025). El umbral se cambia con `UMBRAL_GOLES` en el script.
- **Predictoras**: `RematesTotales`, `RematesAlArco` y `CornersTotales`.
- `TotalGoles` no se usa como predictora porque define el objetivo.
- `PosesionPromedio` **se excluye**: como la posesión del local y la del visitante suman 100, su promedio vale 50 en 1.265 de los 1.266 partidos (en uno vale 50,5). Una variable casi constante no aporta información y, al estandarizarla, amplificaría ruido. Por eso conviene interpretar con cuidado su correlación con los goles en el análisis exploratorio, que depende de ese único partido.

### Metodología
- **Partición temporal**: entrenamiento con las temporadas 2023–2024 (756 partidos) y prueba con 2025 (510 partidos), que el modelo no vio al entrenarse.
- **Supuestos**: variable binaria, muestra suficiente (79 eventos por variable) y ausencia de multicolinealidad.
- **Escalado**: estandarización de las variables dentro de un `Pipeline`, ajustada solo con los datos de entrenamiento de cada partición.
- **Regularización**: se comparan regresión sin regularización, L2 (Ridge), L1 (Lasso) y ElasticNet. El parámetro `C` (inversa de la fuerza de regularización) se elige con validación cruzada estratificada de 5 particiones, minimizando la pérdida logarítmica.
- **Modelo elegido**: el regularizado con mejor desempeño en validación cruzada (decidido solo con datos de entrenamiento): L2 (Ridge), con C = 0,1274.
- **Evaluación**: accuracy, precision, recall, F1, AUC y pérdida logarítmica, comparados con un modelo base que siempre predice la proporción de partidos con muchos goles del entrenamiento. Además se incluyen matriz de confusión, reporte de clasificación, curva ROC, validación cruzada con media y desvío, odds ratios, frontera de decisión y la predicción de un partido promedio.

### Resultados (temporada 2025, umbral 0,5)

| Modelo | Accuracy | Precision | Recall | F1 | AUC | Pérdida logarítmica |
|---|---|---|---|---|---|---|
| Base (proporción de partidos con muchos goles) | 0,676 | 0,000 | 0,000 | 0,000 | 0,500 | 0,630 |
| Sin regularización | 0,692 | 0,643 | 0,109 | 0,187 | 0,715 | 0,579 |
| L2 (Ridge), elegido | 0,692 | 0,654 | 0,103 | 0,178 | 0,713 | 0,580 |
| L1 (Lasso) | 0,692 | 0,643 | 0,109 | 0,187 | 0,715 | 0,579 |
| ElasticNet | 0,692 | 0,654 | 0,103 | 0,178 | 0,714 | 0,579 |
| L2 con `class_weight='balanced'` | 0,700 | 0,542 | 0,473 | 0,505 | 0,710 | 0,608 |

Validación cruzada del modelo elegido: accuracy 0,704 ± 0,038 y AUC 0,725 ± 0,045.

Coeficientes (variables estandarizadas) y odds ratios del modelo elegido:

| Variable | Coeficiente | Odds ratio (por +1 desvío estándar) |
|---|---|---|
| `RematesAlArco` | 0,689 | 1,99 |
| `RematesTotales` | 0,134 | 1,14 |
| `CornersTotales` | −0,184 | 0,83 |

### Observaciones e interpretación
- Los modelos mejoran al modelo base, pero de forma **moderada**: el AUC pasa de 0,50 a 0,71 y la accuracy de 67,6% a 69,2%.
- Los remates al arco son la variable con más peso: un desvío estándar más duplica aproximadamente los odds de que el partido tenga 3 o más goles. Los remates totales aportan poco una vez controlados los remates al arco, y los córners tienen una asociación levemente negativa.
- La regularización casi no modifica los resultados: hay solo 3 variables, sin multicolinealidad y con muchos eventos por variable, por lo que no hay sobreajuste que corregir. L1 no deja ninguna variable en cero.
- Con el umbral de 0,5 el modelo casi no detecta los partidos de muchos goles (recall de 10%) porque son una clase minoritaria. Bajar el umbral (a 0,25 el recall sube a 73%, con precisión de 48%) o ponderar las clases con `class_weight='balanced'` (recall de 47%) mejora la detección a costa de la precisión. La tabla de umbrales es descriptiva: el umbral no se eligió mirando la temporada de prueba. Al ponderar las clases las probabilidades dejan de estar calibradas, por eso empeora la pérdida logarítmica.
- Para un partido con los valores promedio de las variables, el modelo estima un 29,7% de probabilidad de que tenga 3 o más goles.

### Limitaciones
- Remates y córners corresponden al partido ya disputado: el modelo describe qué se asocia con los partidos de muchos goles, **no los predice antes de jugarse**.
- Se trata de asociaciones, no de causalidad.
- La definición del objetivo (3 o más goles) es una decisión de diseño; con otro umbral los resultados cambian.
- La prueba usa una sola temporada, por lo que las diferencias pequeñas entre modelos no son concluyentes.

### Archivos generados (`resultados/regresion_logistica/`)
- `logistica_resumen.txt`: resumen completo en texto.
- `logistica_metricas.csv`, `logistica_coeficientes.csv`, `logistica_odds_ratios.csv`, `logistica_vif.csv`, `logistica_umbral.csv`: tablas.
- `logistica_trayectoria_regularizacion.png`: evolución de los coeficientes según la fuerza de regularización.
- `logistica_roc.png`: curvas ROC de los modelos.
- `logistica_matriz_confusion.png`: matriz de confusión del modelo elegido.
- `logistica_frontera_decision.png`: frontera de decisión según remates al arco y remates totales.
