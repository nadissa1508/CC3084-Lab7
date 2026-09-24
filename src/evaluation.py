"""Visualizacion y analisis de errores (Ejercicio 8)."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import TARGET_COLUMN

sns.set_theme(style="whitegrid")


def add_residuals(predictions: DataFrame, label_col: str = TARGET_COLUMN) -> DataFrame:
    """Residuo = real - predicho: positivo => subestimacion; negativo => sobreestimacion."""
    return predictions.withColumn("residuo", F.col(label_col) - F.col("prediction"))


def sample_for_plots(predictions_with_residuals: DataFrame, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    cols = [TARGET_COLUMN, "prediction", "residuo"]
    total = predictions_with_residuals.count()
    if total <= n:
        return predictions_with_residuals.select(*cols).toPandas()
    fraction = min(1.0, (n * 1.2) / total)
    sample = predictions_with_residuals.select(*cols).sample(
        withReplacement=False, fraction=fraction, seed=seed
    ).limit(n)
    return sample.toPandas()


def plot_actual_vs_predicted(sample_pd: pd.DataFrame, title: str, ax=None):
    ax = ax or plt.gca()
    sns.scatterplot(data=sample_pd, x=TARGET_COLUMN, y="prediction", alpha=0.3, s=15, ax=ax)
    lims = [
        min(sample_pd[TARGET_COLUMN].min(), sample_pd["prediction"].min()),
        max(sample_pd[TARGET_COLUMN].max(), sample_pd["prediction"].max()),
    ]
    ax.plot(lims, lims, color="red", linestyle="--", label="y = x")
    ax.set_title(title)
    ax.set_xlabel("Salario real")
    ax.set_ylabel("Salario predicho")
    ax.legend()
    return ax


def plot_residuals(sample_pd: pd.DataFrame, title: str, ax=None):
    ax = ax or plt.gca()
    sns.scatterplot(data=sample_pd, x="prediction", y="residuo", alpha=0.3, s=15, ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Salario predicho")
    ax.set_ylabel("Residuo (real - predicho)")
    return ax


def error_by_group(predictions_with_residuals: DataFrame, group_col: str) -> pd.DataFrame:
    """MAE y error medio (con signo) por grupo, sobre TODOS los registros de prueba."""
    return (
        predictions_with_residuals.groupBy(group_col)
        .agg(
            F.count("*").alias("n"),
            F.mean(F.abs(F.col("residuo"))).alias("mae"),
            F.mean(F.col("residuo")).alias("error_medio"),
        )
        .orderBy(F.desc("n"))
        .toPandas()
    )
