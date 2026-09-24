# Universidad del Valle de Guatemala  
## Departamento de Ciencias de la Computación  
## CC3084 Data Science 

# Laboratorio 7 — Spark MLlib (ENEIC)

Análisis exploratorio, segmentación (KMeans) y predicción de salario mensual
(regresión lineal vs. Random Forest) sobre la base de Personas de la ENEIC
(INE Guatemala), usando PySpark 3.5 / MLlib.

## Autores

### Cristian Túnchez (231359)  
### Nadissa López (23764)


## Estructura

```
Spark_MLlib/
├── src/                    # módulos reutilizables (import desde el notebook)
│   ├── config.py           # rutas, mapeo de columnas, códigos válidos
│   ├── spark_session.py    # creación de la SparkSession
│   ├── io_utils.py         # lectura de Excel/SPSS -> Spark -> Parquet
│   ├── prepare.py          # armonización, filtros de población, features
│   ├── eda.py               # estadística descriptiva, gráficos, correlación
│   ├── clustering.py       # pipeline KMeans + selección de K
│   ├── modeling.py         # pipelines de regresión (Lineal y Random Forest)
│   └── evaluation.py       # gráficos de error y métricas por grupo
├── notebooks/
│   └── Lab7_Spark_MLlib.ipynb   # notebook orquestador (ejercicios 1-8)
├── data/
│   ├── raw/                # archivos originales ENEIC (NO se versionan/modifican)
│   └── processed/          # Parquet generado + modelos guardados (reproducible)
├── requirements.txt
├── codebook.md              # detalle de las variables utilizadas
├── Dockerfile / docker-compose.yml   # ambiente Spark 3.5 + Jupyter (opcional)
└── README.md
```

## 1. Datos requeridos

Los archivos crudos de la ENEIC (bases de Personas, `.sav`) **no se
versionan** (`data/raw/` está en `.gitignore`) y ya deben colocarse ahí antes
de ejecutar el notebook:

| Archivo en `data/raw/` | periodo_archivo |
|---|---|
| `Personas_ENEIC_T1_2025.sav` | `2025T1` |
| `Personas ENEIC T2-2025.sav` | `2025T2` |
| `Base de datos Personas ENEIC III 2025.sav` | `2025T3` |
| `Base de datos Personas ENEIC IV 2025.sav` | `2025T4` |
| `Base de datos Personas ENEIC I 2026.sav` | `2026T1` |

Junto con los diccionarios de datos (`Diccionario*.xlsx`), usados para
validar los códigos categóricos en [`src/config.py`](src/config.py) (ya
completados para `nivel_educativo` y `dominio`, verificados contra las
etiquetas de valor embebidas en los `.sav` con `pyreadstat`).

`src/io_utils.load_raw_file` acepta `.xlsx`/`.xls` (vía `openpyxl`) y `.sav`
(vía `pyreadstat`). Cada archivo se procesa individualmente para controlar
memoria, y homologa los tipos antes de convertir a Spark.

Ver [`codebook.md`](codebook.md) para el detalle completo de variables y
códigos.

## 2. Ambiente

**Opción A — Docker (recomendado, igual al usado en el curso):**

```bash
docker compose up --build
```

Jupyter queda disponible en `http://localhost:8889` (sin token).

**Opción B — entorno local (Python 3.11 + Java 17):**

```bash
pip install -r requirements.txt
jupyter lab
```

## 3. Ejecución

Abrir [`notebooks/Lab7_Spark_MLlib.ipynb`](notebooks/Lab7_Spark_MLlib.ipynb)
y ejecutar de principio a fin. El notebook no depende de variables creadas
en ejecuciones anteriores: cada sección recarga desde `data/processed/` lo
que generó la sección anterior.

Orden de ejecución:

1. **Carga y armonización** (`src/io_utils.py`, `src/prepare.py`) — genera
   `data/processed/eneic_2025_prepared.parquet` y `eneic_2026_prepared.parquet`.
2. **Estadística descriptiva** (`src/eda.py`).
3. **Correlaciones** (`src/eda.py`).
4. **Clustering KMeans** (`src/clustering.py`).
5. **Regresión lineal** y **Random Forest** (`src/modeling.py`) — guarda el
   mejor modelo de cada algoritmo en `data/processed/models/`.
6. **Evaluación final en 2026T1** y **análisis de errores**
   (`src/evaluation.py`).

## Notas de reproducibilidad

- El clustering, los modelos y las métricas principales son **no
  ponderados** (no usan `FACTOR`); los resultados describen los registros
  analizados, no a la población guatemalteca.
- Todas las métricas (MAE, RMSE, R², tablas por grupo) se calculan sobre el
  conjunto completo correspondiente; solo el graficado punto a punto usa una
  muestra de hasta 5,000 registros.
