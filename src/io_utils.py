"""Lectura de archivos crudos ENEIC (Excel/SPSS) y conversion a Spark/Parquet.

Spark no lee Excel de forma nativa, por lo que cada archivo se carga con
pandas (+openpyxl o +pyreadstat segun la extension), se homologan tipos de
forma explicita y se convierte a Spark DataFrame antes de escribir Parquet.
Cada archivo se procesa de forma individual para controlar memoria.
"""
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import config


def _read_raw_pandas(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".sav":
        import pyreadstat

        df, _meta = pyreadstat.read_sav(str(path))
        return df
    raise ValueError(f"Formato no soportado para {path.name}: {suffix}")


def _select_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona las columnas requeridas, conservando los codigos originales
    (P05D01, P02A03, ...) para que _homogenize_types pueda operar sobre ellos."""
    # Los codigos de columna pueden venir en distinto case entre archivos.
    colmap = {c.upper(): c for c in df.columns}
    missing = [c for c in config.RAW_SELECTED_COLUMNS if c.upper() not in colmap]
    if missing:
        raise KeyError(
            f"Columnas esperadas ausentes en el archivo: {missing}. "
            "Revise el diccionario de datos del trimestre."
        )
    selected = df[[colmap[c.upper()] for c in config.RAW_SELECTED_COLUMNS]].copy()
    selected.columns = config.RAW_SELECTED_COLUMNS
    return selected


def _rename_to_analytic(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el mapeo codigo original -> nombre analitico (config.COLUMN_RENAME)."""
    return df.rename(columns=config.COLUMN_RENAME)


def _homogenize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Homologa tipos antes de crear el Spark DataFrame.

    Un mismo codigo puede llegar como numero o como texto segun el archivo;
    aqui se fuerza una representacion consistente: numericos -> float,
    categoricos de codigo -> string sin decimales espurios (p.ej. "1.0"->"1").
    """
    df = df.copy()
    for col in config.NUMERIC_RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.CATEGORICAL_RAW_COLUMNS:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        df.loc[df[col].isin(["nan", "None", ""]), col] = None
    return df


def load_raw_file(
    spark: SparkSession,
    path: Path,
    periodo_archivo: str,
) -> DataFrame:
    """Carga un archivo crudo y devuelve un Spark DataFrame homologado.

    Agrega las columnas de trazabilidad: archivo_origen, periodo_archivo,
    anio_archivo, trimestre_calendario (a partir del archivo, no de TRIMESTRE).
    """
    if periodo_archivo not in config.FILE_PERIODS:
        raise KeyError(f"periodo_archivo desconocido: {periodo_archivo}")

    pdf = _read_raw_pandas(path)
    pdf = _select_raw_columns(pdf)
    pdf = _homogenize_types(pdf)
    pdf = _rename_to_analytic(pdf)

    sdf = spark.createDataFrame(pdf)

    meta = config.FILE_PERIODS[periodo_archivo]
    sdf = (
        sdf.withColumn("archivo_origen", F.lit(path.name))
        .withColumn("periodo_archivo", F.lit(periodo_archivo))
        .withColumn("anio_archivo", F.lit(meta["anio_archivo"]).cast(T.IntegerType()))
        .withColumn(
            "trimestre_calendario", F.lit(meta["trimestre_calendario"]).cast(T.IntegerType())
        )
    )
    return sdf


def save_parquet(df: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(path))


def read_parquet(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.parquet(str(path))
