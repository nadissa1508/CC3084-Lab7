"""Estadistica descriptiva, graficos exploratorios y correlaciones (Ejercicios 2-3).

Las metricas y estadisticas se calculan siempre sobre el DataFrame completo
de Spark. Solo se transfieren a pandas muestras (<=5000-10000 filas) o
tablas ya agregadas para graficar, segun lo exigido por el enunciado.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

sns.set_theme(style="whitegrid")


def numeric_summary(df: DataFrame, columns: list[str]) -> pd.DataFrame:
    """count, media, mediana, std, min, max, p25, p75, p95 por columna."""
    rows = []
    for c in columns:
        stats = df.select(
            F.count(F.col(c)).alias("n"),
            F.mean(F.col(c)).alias("media"),
            F.stddev(F.col(c)).alias("std"),
            F.min(F.col(c)).alias("min"),
            F.max(F.col(c)).alias("max"),
        ).first()
        p25, mediana, p75, p95 = df.approxQuantile(c, [0.25, 0.5, 0.75, 0.95], 0.001)
        rows.append(
            {
                "variable": c,
                "n": stats["n"],
                "media": stats["media"],
                "mediana": mediana,
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "p25": p25,
                "p75": p75,
                "p95": p95,
            }
        )
    return pd.DataFrame(rows)


def to_pandas_sample(df: DataFrame, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    total = df.count()
    if total <= n:
        return df.toPandas()
    fraction = min(1.0, (n * 1.2) / total)
    sample = df.sample(withReplacement=False, fraction=fraction, seed=seed).limit(n)
    return sample.toPandas()


def category_distribution(df: DataFrame, column: str) -> pd.DataFrame:
    """Distribucion (conteo y %) de una variable categorica, calculada en Spark."""
    counts = df.groupBy(column).count().orderBy(F.desc("count")).toPandas()
    counts["pct"] = 100 * counts["count"] / counts["count"].sum()
    return counts


def plot_category_distribution(df: DataFrame, column: str, title: str | None = None, ax=None):
    data = category_distribution(df, column)
    ax = ax or plt.gca()
    sns.barplot(data=data, x=column, y="count", ax=ax)
    ax.set_title(title or f"Distribucion de {column}")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_salary_distribution(df: DataFrame, log_scale: bool = False, sample_n: int = 5000, ax=None):
    sample = to_pandas_sample(df.select("salario_mensual"), n=sample_n)
    ax = ax or plt.gca()
    sns.histplot(sample["salario_mensual"], kde=True, ax=ax)
    if log_scale:
        ax.set_xscale("log")
    ax.set_title("Distribucion de salario_mensual" + (" (escala log)" if log_scale else ""))
    ax.set_xlabel("salario_mensual (Q)")
    return ax


def median_by_group(df: DataFrame, group_col: str, value_col: str = "salario_mensual") -> pd.DataFrame:
    return (
        df.groupBy(group_col)
        .agg(
            F.count("*").alias("n"),
            F.expr(f"percentile_approx({value_col}, 0.5)").alias("mediana"),
            F.mean(value_col).alias("media"),
        )
        .orderBy(F.desc("mediana"))
        .toPandas()
    )


def median_salary_by_period(df: DataFrame) -> pd.DataFrame:
    return (
        df.groupBy("periodo_archivo")
        .agg(
            F.count("*").alias("n"),
            F.expr("percentile_approx(salario_mensual, 0.5)").alias("mediana_salario"),
        )
        .orderBy("periodo_archivo")
        .toPandas()
    )


def correlation_matrix(df: DataFrame, columns: list[str]):
    """Matriz de correlacion de Pearson via VectorAssembler + Correlation.corr."""
    assembler = VectorAssembler(inputCols=columns, outputCol="_features_corr")
    vectorized = assembler.transform(df.select(*columns).na.drop()).select("_features_corr")
    corr = Correlation.corr(vectorized, "_features_corr", "pearson").head()[0]
    corr_pd = pd.DataFrame(corr.toArray(), index=columns, columns=columns)
    return corr_pd


def plot_correlation_heatmap(corr_pd: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    sns.heatmap(corr_pd, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de correlacion (Pearson)")
    return ax
