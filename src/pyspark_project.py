import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# 1. WINDOWS-Ի ԵՎ JAVA-Ի ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
os.environ['HADOOP_HOME'] = sys.prefix

def main():
    # 2. ՖԱՅԼԻ ՃԱՆԱՊԱՐՀԻ ՈՐՈՇՈՒՄ (Ճիշտ տարբերակ .py ֆայլի համար)
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "Student_data.csv"
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Չհաջողվեց գտնել ֆայլը այստեղ: {DATA_PATH}")

    # 3. SPARK ՍԵՍԻԱՅԻ ՍՏԵՂԾՈՒՄ
    spark = SparkSession.builder \
        .appName("StudentCGPAPrediction") \
        .master("local[*]") \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # 4. ՏՎՅԱԼՆԵՐԻ ԲԵՌՆՈՒՄ
        df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
        
        # Սյունակների սահմանում ըստ ձեր CSV ֆայլի
        label_col = "Final_CGPA"
        feature_cols = ["Attendance_Pct", "Study_Hours_Per_Day", "Previous_CGPA"]

        # 5. ՄՇԱԿՄԱՆ ՓՈՒԼԵՐ
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
        lr = LinearRegression(featuresCol="features", labelCol=label_col)

        pipeline = Pipeline(stages=[assembler, scaler, lr])

        # 6. ՏՎՅԱԼՆԵՐԻ ԲԱԺԱՆՈՒՄ (80% / 20%)
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        
        # 7. ՄԱՐԶՈՒՄ ԵՎ ԿԱՆԽԱՏԵՍՈՒՄ
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        
        print("\n" + "="*50)
        print("ԿԱՆԽԱՏԵՍՎԱԾ ՄՈԳ-ԵՐԻ ՆՄՈՒՇ (ԱՌԱՋԻՆ 5 ՏՈՂ):")
        print("="*50)
        predictions.select(label_col, "prediction").show(5)

        # 8. ՈՐԱԿԻ ԳՆԱՀԱՏՈՒՄ
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")
        r2 = evaluator.evaluate(predictions)
        print(f"R2 Score = {r2:.4f}\n")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
    