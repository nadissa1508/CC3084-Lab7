"""Creacion de la SparkSession usada en todo el laboratorio."""
from pyspark.sql import SparkSession


def get_spark(app_name: str = "Lab7-Spark-MLlib") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
