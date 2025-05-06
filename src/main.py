import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when, rand
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkConf, SparkContext
from prettytable import PrettyTable

conf = SparkConf().setAppName("SentimentAnalysisMP2L").setMaster("local[*]") \
    .set("spark.executor.memory", "4g") \
    .set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)

spark = SparkSession.builder.appName("SentimentAnalysisMP2L").getOrCreate()

df = spark.read.csv("data/avis_etudiants_dataset.csv", header=True, inferSchema=True)
df = df.select("Text", "Matière", "Semestre").na.drop()

df = df.withColumn("Sentiment",
    when(lower(col("Text")).rlike(".*(pas|nul|difficile|compliqué|mauvais|incompréhensible).*"), "negative")
    .when(lower(col("Text")).rlike(".*(bon|utile|clair|intéressant|super|excellent|parfait|facile|bien|génial).*"), "positive")
    .otherwise("neutral")
)

df = df.withColumn("Sentiment", 
    when((col("Sentiment") == "neutral") & (rand() < 0.9), "positive")  # 90% des neutres deviennent positifs
    .otherwise(col("Sentiment"))
)

indexer = StringIndexer(inputCol="Sentiment", outputCol="label")

tokenizer = Tokenizer(inputCol="Text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[indexer, tokenizer, remover, hashingTF, idf, lr])

training, test = df.randomSplit([0.7, 0.3])

model = pipeline.fit(training)

predictions = model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

table = PrettyTable()
table.field_names = ["📝 Avis", "🎯 Réel", "🤖 Prédit"]

labels_mapping = indexer.fit(df).labels
labels_dict = {i: label for i, label in enumerate(labels_mapping)}

rows = predictions.select("Text", "Sentiment", "prediction").take(15)
for row in rows:
    texte = row.Text[:60].replace("\n", " ") + "..."
    sentiment = row.Sentiment
    prediction = labels_dict[int(row.prediction)] if int(row.prediction) in labels_dict else "?"
    table.add_row([texte, sentiment, prediction])

print("\n📊 Exemples de prédictions :")
print(table)

sentiment_counts = predictions.groupBy("Sentiment").count().collect()
sentiment_map = {"negative": 0, "neutral": 0, "positive": 0}
for row in sentiment_counts:
    sentiment_map[row["Sentiment"]] = row["count"]

labels = ["Négatif", "Neutre", "Positif"]
values = [sentiment_map["negative"], sentiment_map["neutral"], sentiment_map["positive"]]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=["#001f3f", "#0074D9", "#00BFFF"])  # Bleu foncé → clair
plt.title("Répartition des sentiments")
plt.xlabel("Catégorie")
plt.ylabel("Nombre d'avis")
plt.savefig('output/results.png')
plt.show()

print(f"\n✅ Analyse terminée")
print(f"- Données d'entraînement : {training.count()} exemples")
print(f"- Données de test : {test.count()} exemples")
print(f"- Précision du modèle : {accuracy:.2f}")

# interface web
def get_analysis_results():
    return {
        "accuracy": accuracy,
        "counts": sentiment_map,
        "examples": rows
    }

if __name__ == "__main__":
    results = get_analysis_results()
    print(f"Analyse terminée avec précision: {results['accuracy']:.2f}")
    spark.stop()
