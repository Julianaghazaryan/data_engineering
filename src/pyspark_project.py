import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# ---------------------------------------------------
# WINDOWS & JAVA CONFIGURATION
# ---------------------------------------------------

os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
os.environ['HADOOP_HOME'] = sys.prefix


def main():

    # ---------------------------------------------------
    # DATA PATH
    # ---------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "Student_data.csv"

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    # ---------------------------------------------------
    # CREATE SPARK SESSION
    # ---------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("Student_CGPA_Prediction")
        .master("local[*]")
        .config("spark.driver.host", "localhost")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:

        # ---------------------------------------------------
        # LOAD DATASET
        # ---------------------------------------------------

        df = spark.read.csv(
            str(DATA_PATH),
            header=True,
            inferSchema=True
        )

        print("\nDataset Preview:")
        df.show(5)

        print("\nDataset Schema:")
        df.printSchema()

        # ---------------------------------------------------
        # HANDLE MISSING VALUES
        # ---------------------------------------------------

        df = df.dropna()

        # ---------------------------------------------------
        # FEATURE & LABEL COLUMNS
        # ---------------------------------------------------

        label_col = "Final_CGPA"

        feature_cols = [
            "Attendance_Pct",
            "Study_Hours_Per_Day",
            "Previous_CGPA"
        ]

        # ---------------------------------------------------
        # FEATURE ENGINEERING
        # ---------------------------------------------------

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="raw_features"
        )

        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withStd=True,
            withMean=False
        )

        # ---------------------------------------------------
        # MODEL
        # ---------------------------------------------------

        lr = LinearRegression(
            featuresCol="features",
            labelCol=label_col
        )

        # ---------------------------------------------------
        # PIPELINE
        # ---------------------------------------------------

        pipeline = Pipeline(
            stages=[assembler, scaler, lr]
        )

        # ---------------------------------------------------
        # TRAIN / TEST SPLIT
        # ---------------------------------------------------

        train_data, test_data = df.randomSplit(
            [0.8, 0.2],
            seed=42
        )

        print(f"\nTraining Rows: {train_data.count()}")
        print(f"Testing Rows: {test_data.count()}")

        # ---------------------------------------------------
        # TRAIN MODEL
        # ---------------------------------------------------

        model = pipeline.fit(train_data)

        # ---------------------------------------------------
        # MAKE PREDICTIONS
        # ---------------------------------------------------

        predictions = model.transform(test_data)

        print("\nPredictions Sample:")
        predictions.select(
            label_col,
            "prediction"
        ).show(5)

        # ---------------------------------------------------
        # EVALUATION
        # ---------------------------------------------------

        r2_evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="r2"
        )

        rmse_evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="rmse"
        )

        r2 = r2_evaluator.evaluate(predictions)
        rmse = rmse_evaluator.evaluate(predictions)

        print("\nModel Evaluation:")
        print(f"R2 Score  : {r2:.4f}")
        print(f"RMSE Score: {rmse:.4f}")

    finally:

        # ---------------------------------------------------
        # STOP SPARK SESSION
        # ---------------------------------------------------

        spark.stop()


if __name__ == "__main__":
    main()
