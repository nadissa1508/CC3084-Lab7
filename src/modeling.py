"""Pipelines de regresion supervisada: Lineal y Random Forest (Ejercicios 5-7)."""
from dataclasses import dataclass, field

import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.sql import DataFrame

from .config import MODEL_CATEGORICAL_PREDICTORS, MODEL_NUMERIC_PREDICTORS, RANDOM_SEED, TARGET_COLUMN


def _categorical_encoding_stages(categorical_cols: list[str]):
    indexers = [
        StringIndexer(
            inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"
        )
        for c in categorical_cols
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_cols],
        outputCols=[f"{c}_ohe" for c in categorical_cols],
    )
    return indexers, encoder


def build_linear_regression_pipeline(
    reg_param: float,
    elastic_net_param: float,
    numeric_cols: list[str] = MODEL_NUMERIC_PREDICTORS,
    categorical_cols: list[str] = MODEL_CATEGORICAL_PREDICTORS,
) -> Pipeline:
    indexers, encoder = _categorical_encoding_stages(categorical_cols)
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{c}_ohe" for c in categorical_cols],
        outputCol="features",
    )
    lr = LinearRegression(
        featuresCol="features",
        labelCol=TARGET_COLUMN,
        regParam=reg_param,
        elasticNetParam=elastic_net_param,
        standardization=True,  # estandarizacion interna de los predictores
    )
    return Pipeline(stages=[*indexers, encoder, assembler, lr])


def build_random_forest_pipeline(
    num_trees: int,
    max_depth: int,
    numeric_cols: list[str] = MODEL_NUMERIC_PREDICTORS,
    categorical_cols: list[str] = MODEL_CATEGORICAL_PREDICTORS,
) -> Pipeline:
    indexers, encoder = _categorical_encoding_stages(categorical_cols)
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{c}_ohe" for c in categorical_cols],
        outputCol="features",
    )
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=TARGET_COLUMN,
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=RANDOM_SEED,
    )
    return Pipeline(stages=[*indexers, encoder, assembler, rf])


def evaluate_predictions(predictions: DataFrame, label_col: str = TARGET_COLUMN) -> dict:
    metrics = {}
    for metric_name in ("mae", "rmse", "r2"):
        evaluator = RegressionEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName=metric_name
        )
        metrics[metric_name] = evaluator.evaluate(predictions)
    return metrics


@dataclass
class ModelRun:
    name: str
    params: dict
    model: PipelineModel = field(repr=False)
    metrics: dict


def run_configs(
    pipelines_with_meta: list[tuple[str, dict, Pipeline]],
    train_df: DataFrame,
    val_df: DataFrame,
) -> tuple[list[ModelRun], ModelRun]:
    """Ajusta cada pipeline con train, evalua con val y regresa (runs, mejor_run).

    El mejor se selecciona por menor RMSE de validacion.
    """
    runs: list[ModelRun] = []
    for name, params, pipeline in pipelines_with_meta:
        fitted = pipeline.fit(train_df)
        preds = fitted.transform(val_df)
        metrics = evaluate_predictions(preds)
        runs.append(ModelRun(name=name, params=params, model=fitted, metrics=metrics))
    best = min(runs, key=lambda r: r.metrics["rmse"])
    return runs, best


def runs_to_table(runs: list[ModelRun]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"config": r.name, **r.params, **r.metrics} for r in runs]
    )
