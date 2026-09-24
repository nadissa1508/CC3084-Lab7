"""Configuracion centralizada del laboratorio: rutas, columnas y codigos.

Los valores marcados con TODO dependen del diccionario de datos oficial de
cada archivo ENEIC (INE) y deben completarse cuando los archivos crudos
esten disponibles en data/raw/.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"

TRAIN_2025_PARQUET = PROCESSED_DIR / "eneic_2025_prepared.parquet"
TEST_2026_PARQUET = PROCESSED_DIR / "eneic_2026_prepared.parquet"

# ---------------------------------------------------------------------------
# Columnas originales -> nombre analitico (Tabla "Variables que utilizara")
# ---------------------------------------------------------------------------
COLUMN_RENAME = {
    "P05D01": "salario_mensual",
    "P02A03": "edad",
    "P05C07A": "antiguedad_anios_raw",
    "P05C07B": "antiguedad_meses_raw",
    "P05H01A": "horas_semanales",
    "P03A03A": "nivel_educativo",
    "P05C16": "categoria_ocupacional",
    "DOMINIO": "dominio",
    "OCUPADOS": "ocupado",
    # Se conservan con su nombre original (auditoria / diseno muestral)
    "NUM_HOGAR": "NUM_HOGAR",
    "NUM_PERSONA": "NUM_PERSONA",
    "FACTOR": "FACTOR",
    "ANIO": "ANIO",
    "TRIMESTRE": "TRIMESTRE",
}

# Columnas originales que se deben seleccionar de cada archivo crudo
RAW_SELECTED_COLUMNS = list(COLUMN_RENAME.keys())

# Tipos explicitos esperados tras la conversion (antes del cast final en Spark)
NUMERIC_RAW_COLUMNS = [
    "P05D01", "P02A03", "P05C07A", "P05C07B", "P05H01A",
    "NUM_HOGAR", "NUM_PERSONA", "FACTOR", "ANIO", "TRIMESTRE", "OCUPADOS",
]
CATEGORICAL_RAW_COLUMNS = ["P03A03A", "P05C16", "DOMINIO"]

# ---------------------------------------------------------------------------
# Identificacion del periodo por archivo de procedencia
# Ver seccion "IDENTIFICACION DEL PERIODO" del enunciado.
# Cada archivo crudo debe cargarse indicando explicitamente su periodo;
# NO se infiere restando 1 a TRIMESTRE.
# ---------------------------------------------------------------------------
FILE_PERIODS = {
    "2025T1": {"anio_archivo": 2025, "trimestre_calendario": 1, "uso": "train"},
    "2025T2": {"anio_archivo": 2025, "trimestre_calendario": 2, "uso": "train"},
    "2025T3": {"anio_archivo": 2025, "trimestre_calendario": 3, "uso": "train"},
    "2025T4": {"anio_archivo": 2025, "trimestre_calendario": 4, "uso": "train_final"},
    "2026T1": {"anio_archivo": 2026, "trimestre_calendario": 1, "uso": "test"},
}

# ---------------------------------------------------------------------------
# Codigos validos de variables categoricas (Diccionario de datos ENEIC/INE)
# ---------------------------------------------------------------------------
CATEGORIA_OCUPACIONAL_LABELS = {
    "1": "Empleado de gobierno",
    "2": "Empleado de empresa privada",
    "3": "Empleado jornalero o peon",
    "4": "Servicio domestico",
}
CATEGORIA_OCUPACIONAL_VALID_CODES = list(CATEGORIA_OCUPACIONAL_LABELS.keys())

# Fuente: etiquetas de valor embebidas en los .sav (consistentes en las 5
# bases, verificado con pyreadstat). El codigo 0 = "ninguno" es valido y NO
# debe tratarse como faltante.
NIVEL_EDUCATIVO_LABELS: dict[str, str] = {
    "0": "NINGUNO",
    "1": "PREPRIMARIA",
    "2": "PRIMARIA",
    "3": "BASICO",
    "4": "DIVERSIFICADO",
    "5": "SUPERIOR",
    "6": "MAESTRIA",
    "7": "DOCTORADO",
}
NIVEL_EDUCATIVO_VALID_CODES: list[str] | None = list(NIVEL_EDUCATIVO_LABELS.keys())

# DOMINIO: dominio de estudio (codigos geograficos), fuente: .sav (idem).
DOMINIO_LABELS: dict[str, str] = {
    "1": "Urbano Metropolitano",
    "2": "Resto Urbano",
    "3": "Rural Nacional",
}
DOMINIO_VALID_CODES: list[str] | None = list(DOMINIO_LABELS.keys())

MISSING_LABEL = "DESCONOCIDO"

# ---------------------------------------------------------------------------
# Poblacion analitica y predictores del modelo
# ---------------------------------------------------------------------------
MIN_EDAD = 15
MAX_HORAS_SEMANALES = 168
KEY_COLUMNS = ["periodo_archivo", "NUM_HOGAR", "NUM_PERSONA"]

MODEL_NUMERIC_PREDICTORS = ["edad", "antiguedad", "horas_semanales"]
MODEL_CATEGORICAL_PREDICTORS = ["nivel_educativo", "categoria_ocupacional", "dominio"]
MODEL_PREDICTORS = MODEL_NUMERIC_PREDICTORS + MODEL_CATEGORICAL_PREDICTORS
TARGET_COLUMN = "salario_mensual"

RANDOM_SEED = 42
