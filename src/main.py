import matplotlib.pyplot as plt
import json
from pathlib import Path
from prettytable import PrettyTable
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 🔧 Configuration Spark
conf = SparkConf().setAppName("SentimentAnalysisMP2L").setMaster("local[*]") \
    .set("spark.executor.memory", "4g") \
    .set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# 📁 Définition des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "avis_etudiants_dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 📄 Chargement des données CSV
df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
df = df.select("Text", "Matière", "Semestre", "Annee").na.drop()

# 🧹 Nettoyage + Labellisation automatique
df = df.withColumn("Sentiment",
    when(lower(col("Text")).rlike(".*(pas|nul|difficile|compliqué|mauvais|incompréhensible).*"), "negative")
    .when(lower(col("Text")).rlike(".*(bon|utile|clair|intéressant|super|excellent|parfait|facile|bien|génial).*"), "positive")
    .otherwise("neutral")
)

# 🛠️ Création du pipeline NLP + Classification
indexer = StringIndexer(inputCol="Sentiment", outputCol="label")
tokenizer = Tokenizer(inputCol="Text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)

pipeline = Pipeline(stages=[indexer, tokenizer, remover, hashingTF, idf, lr])

# 🧪 Split des données
train, test = df.randomSplit([0.8, 0.2])
model = pipeline.fit(train)

# 🤖 Prédictions
predictions = model.transform(test)

# 📊 Évaluation du modèle
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# 📝 Affichage d'exemples sous forme de tableau
table = PrettyTable()
table.field_names = ["📝 Avis", "🎯 Réel", "🤖 Prédit"]

labels_mapping = indexer.fit(df).labels
labels_dict = {i: label for i, label in enumerate(labels_mapping)}

rows = predictions.select("Text", "Sentiment", "prediction").take(15)
for row in rows:
    texte = row.Text[:60].replace("\n", " ") + "..."
    sentiment = row.Sentiment
    prediction = labels_dict.get(int(row.prediction), "?")
    table.add_row([texte, sentiment, prediction])

print("\n📊 Exemples de prédictions :")
print(table)

# 💾 Enregistrement du tableau dans un fichier texte
result_txt_path = OUTPUT_DIR / "results.txt"
with open(result_txt_path, "w", encoding="utf-8") as f:
    f.write("📊 Exemples de prédictions :\n")
    f.write(str(table))

print(f"📄 Tableau des prédictions enregistré dans : {result_txt_path}")

# 📊 Graphique : répartition des sentiments
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
print(f"📁 Tableau enregistré : {OUTPUT_DIR / 'results.txt'}")


# 📈 Regroupement des sentiments par année
par_annee = predictions.groupBy("Annee", "Sentiment").count().collect()
grouped_data = {}
for row in par_annee:
    annee = str(row["Annee"])
    sentiment = row["Sentiment"]
    count = row["count"]
    grouped_data.setdefault(annee, {})[sentiment] = count

# 💾 Sauvegarde en JSON
json_path = OUTPUT_DIR / "sentiments_par_annee.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(grouped_data, f, ensure_ascii=False, indent=2)

print(f"📄 Données par année enregistrées dans : {json_path}")

# 🛑 Fermeture de Spark
spark.stop()
