import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when, rand
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkConf, SparkContext
from prettytable import PrettyTable

import json
from pathlib import Path

# Configuration Spark
conf = SparkConf().setAppName("SentimentAnalysisMP2L").setMaster("local[*]") \
    .set("spark.executor.memory", "4g") \
    .set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# Dossiers
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "avis_etudiants_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Charger les données
df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
df = df.select("Text", "Matière", "Semestre", "Annee").na.drop()

# Nettoyage et labellisation manuelle
df = df.withColumn("Sentiment",
    when(lower(col("Text")).rlike(".*(pas|nul|difficile|compliqué|mauvais|incompréhensible).*"), "negative")
    .when(lower(col("Text")).rlike(".*(bon|utile|clair|intéressant|super|excellent|parfait|facile|bien|génial).*"), "positive")
    .otherwise("neutral")
)



# Pipeline NLP + Classification
indexer = StringIndexer(inputCol="Sentiment", outputCol="label")
tokenizer = Tokenizer(inputCol="Text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)

pipeline = Pipeline(stages=[indexer, tokenizer, remover, hashingTF, idf, lr])
train, test = df.randomSplit([0.7, 0.3])
model = pipeline.fit(train)

# Prédictions
predictions = model.transform(test)

# Évaluation
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Exemple d'affichage
table = PrettyTable()
table.field_names = ["📝 Avis", "🎯 Réel", "🤖 Prédit"]
labels_mapping = indexer.fit(df).labels
labels_dict = {i: label for i, label in enumerate(labels_mapping)}

rows = predictions.select("Text", "Sentiment", "prediction").take(10)
for row in rows:
    texte = row.Text[:60].replace("\n", " ") + "..."
    sentiment = row.Sentiment
    prediction = labels_dict[int(row.prediction)] if int(row.prediction) in labels_dict else "?"
    table.add_row([texte, sentiment, prediction])

print("\n📊 Exemples de prédictions :")
print(table)

# Graphique global
sentiment_counts = predictions.groupBy("Sentiment").count().collect()
sentiment_map = {"negative": 0, "neutral": 0, "positive": 0}
for row in sentiment_counts:
    sentiment_map[row["Sentiment"]] = row["count"]

labels = ["Négatif", "Neutre", "Positif"]
values = [sentiment_map["negative"], sentiment_map["neutral"], sentiment_map["positive"]]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=["#001f3f", "#0074D9", "#00BFFF"])
plt.title("Répartition des sentiments (tous avis)")
plt.xlabel("Sentiment")
plt.ylabel("Nombre d'avis")
plt.savefig(OUTPUT_DIR / "results.png")
plt.close()

print(f"\n✅ Précision du modèle : {accuracy:.2f}")
print(f"📁 Graphique enregistré : {OUTPUT_DIR / 'results.png'}")

# Sauvegarde des prédictions par année
par_annee = predictions.groupBy("Annee", "Sentiment").count().collect()
grouped_data = {}
for row in par_annee:
    annee = str(row["Annee"])
    sentiment = row["Sentiment"]
    count = row["count"]
    if annee not in grouped_data:
        grouped_data[annee] = {}
    grouped_data[annee][sentiment] = count

# Enregistrement dans un fichier JSON
json_path = OUTPUT_DIR / "sentiments_par_annee.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(grouped_data, f, ensure_ascii=False, indent=2)

print(f"📄 Données par année enregistrées dans : {json_path}")

spark.stop()
