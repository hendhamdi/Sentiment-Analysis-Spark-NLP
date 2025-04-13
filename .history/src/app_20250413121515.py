from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import Row

app = Flask(__name__)

# Lancer une session Spark
spark = SparkSession.builder.appName("SentimentWebApp").getOrCreate()

# Charger le modèle entraîné (assure-toi qu'il est bien sauvegardé là)
model.write().overwrite().save("output/model")


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_text = request.form['text']
        df = spark.createDataFrame([Row(Text=user_text)])
        prediction = model.transform(df)
        pred_value = prediction.select("prediction").collect()[0][0]

        sentiment_map = {0.0: "Négatif", 1.0: "Neutre", 2.0: "Positif"}
        result = sentiment_map.get(pred_value, "Inconnu")

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
