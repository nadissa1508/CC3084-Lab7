"""Armonizacion, poblacion analitica y construccion de variables (Ejercicio 1).

Todas las funciones son puras (reciben y devuelven Spark DataFrames) para
poder encadenarlas en el notebook y para poder probarlas de forma aislada.
"""
from typing import Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from . import config


def cast_final_types(df: DataFrame) -> DataFrame:
    """Aplica los tipos finales explicitos a las columnas seleccionadas."""
    return (
        df.withColumn("salario_mensual", F.col("salario_mensual").cast(T.DoubleType()))
        .withColumn("edad", F.col("edad").cast(T.DoubleType()))
        .withColumn("antiguedad_anios_raw", F.col("antiguedad_anios_raw").cast(T.DoubleType()))
        .withColumn("antiguedad_meses_raw", F.col("antiguedad_meses_raw").cast(T.DoubleType()))
        .withColumn("horas_semanales", F.col("horas_semanales").cast(T.DoubleType()))
        .withColumn("ocupado", F.col("ocupado").cast(T.IntegerType()))
        .withColumn("NUM_HOGAR", F.col("NUM_HOGAR").cast(T.LongType()))
        .withColumn("NUM_PERSONA", F.col("NUM_PERSONA").cast(T.LongType()))
        .withColumn("FACTOR", F.col("FACTOR").cast(T.DoubleType()))
        .withColumn("ANIO", F.col("ANIO").cast(T.IntegerType()))
        .withColumn("TRIMESTRE", F.col("TRIMESTRE").cast(T.IntegerType()))
    )


def _validate_categorical(
    df: DataFrame, column: str, valid_codes: Optional[list[str]]
) -> DataFrame:
    """Homologa un codigo categorico: nulo/no reconocido -> DESCONOCIDO.

    El codigo "0" (p.ej. nivel educativo "ninguno") es valido y NO se toca.
    """
    col = F.col(column)
    if valid_codes is None:
        return df.withColumn(
            column, F.when(col.isNull() | (col == ""), config.MISSING_LABEL).otherwise(col)
        )
    return df.withColumn(
        column,
        F.when(col.isin(valid_codes), col).otherwise(F.lit(config.MISSING_LABEL)),
    )


def apply_categorical_validation(df: DataFrame) -> DataFrame:
    df = _validate_categorical(df, "nivel_educativo", config.NIVEL_EDUCATIVO_VALID_CODES)
    df = _validate_categorical(df, "dominio", config.DOMINIO_VALID_CODES)
    # categoria_ocupacional se valida en el filtro de poblacion (asalariados);
    # aqui solo se homologan nulos fuera de esos 4 codigos a DESCONOCIDO para
    # los registros que pasen a describir el universo previo al filtro.
    df = _validate_categorical(
        df, "categoria_ocupacional", None
    )
    return df


def build_features(df: DataFrame) -> DataFrame:
    """Construye antiguedad (anios) = antiguedad_anios_raw + antiguedad_meses_raw/12."""
    return df.withColumn(
        "antiguedad",
        F.col("antiguedad_anios_raw") + (F.col("antiguedad_meses_raw") / F.lit(12.0)),
    )


def missingness_report(df: DataFrame, columns: list[str]):
    """Cantidad y % de faltantes por columna, ANTES de aplicar filtros."""
    total = df.count()
    rows = []
    for c in columns:
        n_missing = df.filter(F.col(c).isNull()).count()
        rows.append(
            {
                "variable": c,
                "n_faltantes": n_missing,
                "pct_faltantes": round(100.0 * n_missing / total, 2) if total else None,
            }
        )
    return rows, total


# Orden fijo de los filtros de poblacion analitica + criterios numericos.
# (nombre, condicion) -- se evalua en este orden sobre el dataframe corriente.
FILTER_STEPS = [
    ("ocupado_es_1", lambda df: F.col("ocupado") == 1),
    (
        "asalariado_categoria_valida",
        lambda df: F.col("categoria_ocupacional").isin(
            config.CATEGORIA_OCUPACIONAL_VALID_CODES
        ),
    ),
    (
        "salario_valido_positivo",
        lambda df: F.col("salario_mensual").isNotNull()
        & ~F.isnan(F.col("salario_mensual"))
        & (F.col("salario_mensual") > 0),
    ),
    (
        "edad_valida",
        lambda df: F.col("edad").isNotNull() & ~F.isnan(F.col("edad")) & (F.col("edad") >= config.MIN_EDAD),
    ),
    (
        "antiguedad_anios_no_negativa",
        lambda df: F.col("antiguedad_anios_raw").isNotNull() & (F.col("antiguedad_anios_raw") >= 0),
    ),
    (
        "antiguedad_meses_valido",
        lambda df: F.col("antiguedad_meses_raw").isNotNull()
        & (F.col("antiguedad_meses_raw") >= 0)
        & (F.col("antiguedad_meses_raw") <= 11),
    ),
    ("antiguedad_menor_igual_edad", lambda df: F.col("antiguedad") <= F.col("edad")),
    (
        "horas_semanales_validas",
        lambda df: F.col("horas_semanales").isNotNull()
        & (F.col("horas_semanales") > 0)
        & (F.col("horas_semanales") <= config.MAX_HORAS_SEMANALES),
    ),
]


def apply_population_filters(df: DataFrame):
    """Aplica los filtros en orden fijo y devuelve (df_filtrado, reporte).

    El reporte documenta cuantos registros se excluyen en cada paso, sobre
    el remanente del paso anterior (para que la suma de exclusiones + el
    conteo final coincida con el conteo inicial).
    """
    current = df.cache()
    report = []
    n_inicial = current.count()
    n_prev = n_inicial
    for name, cond_fn in FILTER_STEPS:
        previous = current
        current = current.filter(cond_fn(current)).cache()
        n_after = current.count()
        previous.unpersist()
        report.append(
            {
                "paso": name,
                "registros_excluidos": n_prev - n_after,
                "registros_restantes": n_after,
            }
        )
        n_prev = n_after
    report_summary = {
        "n_inicial": n_inicial,
        "n_final": n_prev,
        "pasos": report,
    }
    return current, report_summary


def check_uniqueness(df: DataFrame, key_columns: list[str] = config.KEY_COLUMNS):
    """Verifica unicidad de la llave y separa duplicados exactos de conflictivos."""
    dup_keys = (
        df.groupBy(*key_columns)
        .count()
        .filter(F.col("count") > 1)
    )
    n_dup_keys = dup_keys.count()
    if n_dup_keys == 0:
        return {"n_llaves_duplicadas": 0, "n_duplicados_exactos": 0, "n_conflictivos": 0}

    dup_records = df.join(dup_keys.select(*key_columns), on=key_columns, how="inner")
    other_cols = [c for c in df.columns if c not in key_columns]
    n_distinct_full = dup_records.select(*key_columns, *other_cols).distinct().count()
    n_distinct_keys_only = dup_records.select(*key_columns).distinct().count()
    # Si el numero de filas completamente distintas == numero de filas totales
    # duplicadas por llave, son conflictos reales (misma llave, distinto contenido).
    n_dup_rows_total = dup_records.count()
    n_exact_duplicate_rows = n_dup_rows_total - n_distinct_full
    n_conflictivos = n_distinct_full - n_distinct_keys_only
    return {
        "n_llaves_duplicadas": n_dup_keys,
        "n_filas_involucradas": n_dup_rows_total,
        "n_duplicados_exactos": n_exact_duplicate_rows,
        "n_conflictivos": n_conflictivos,
    }


def prepare_dataframe(df: DataFrame):
    """Pipeline completo del Ejercicio 1 sobre un DataFrame ya unido (unionByName).

    Devuelve (df_poblacion_analitica, reporte_filtros, reporte_faltantes,
    reporte_unicidad). El reporte de faltantes se calcula ANTES de filtrar.
    """
    df = cast_final_types(df)
    df = build_features(df)
    df = apply_categorical_validation(df).cache()

    missing_cols = config.MODEL_PREDICTORS + [config.TARGET_COLUMN]
    missing_report, _ = missingness_report(df, missing_cols)

    uniqueness_report = check_uniqueness(df, config.KEY_COLUMNS)

    df_filtered, filter_report = apply_population_filters(df)
    df.unpersist()

    return df_filtered, filter_report, missing_report, uniqueness_report
