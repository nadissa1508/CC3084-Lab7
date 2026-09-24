"""Segmentacion de perfiles con KMeans (Ejercicio 4)."""
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import RANDOM_SEED


def build_clustering_pipeline(feature_columns: list[str], k: int) -> Pipeline:
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw", outputCol="features_scaled", withMean=True, withStd=True
    )
    kmeans = KMeans(
        featuresCol="features_scaled",
        predictionCol="cluster",
        k=k,
        seed=RANDOM_SEED,
    )
    return Pipeline(stages=[assembler, scaler, kmeans])


def evaluate_k_candidates(
    df: DataFrame, feature_columns: list[str], k_values: list[int] = (2, 3, 4, 5)
) -> pd.DataFrame:
    """Ajusta un modelo por cada K y calcula el silhouette score (cosine/euclidean)."""
    rows = []
    fitted_models: dict[int, PipelineModel] = {}
    for k in k_values:
        pipeline = build_clustering_pipeline(feature_columns, k)
        model = pipeline.fit(df)
        predictions = model.transform(df)
        evaluator = ClusteringEvaluator(
            featuresCol="features_scaled", predictionCol="cluster", metricName="silhouette"
        )
        silhouette = evaluator.evaluate(predictions)
        rows.append({"k": k, "silhouette": silhouette})
        fitted_models[k] = model
    return pd.DataFrame(rows), fitted_models


def cluster_profile(df_with_clusters: DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Promedio de cada variable por cluster, para describir los perfiles."""
    agg_exprs = [F.count("*").alias("n")] + [
        F.avg(c).alias(f"avg_{c}") for c in feature_columns
    ]
    profile = df_with_clusters.groupBy("cluster").agg(*agg_exprs).toPandas()
    return profile.sort_values("cluster").reset_index(drop=True)
