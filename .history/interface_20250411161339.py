import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import Row

# Charger la session Spark
spark = SparkSession.builder.appName("TestSentiment").getOrCreate()

# Charger le modèle sauvegardé
model = PipelineModel.load("output/model")

st.title("💬 Analyse de sentiment en ligne")

user_input = st.text_area("Entrez un texte à analyser :", "")

if st.button("Analyser"):
    if user_input:
        df = spark.createDataFrame([Row(Text=user_input)])
        prediction = model.transform(df)
        sentiment_index = prediction.select("prediction").collect()[0][0]
        sentiment_map = {0.0: "Négatif", 1.0: "Neutre", 2.0: "Positif"}
        st.success(f"🧠 Sentiment prédit : **{sentiment_map.get(sentiment_index, 'Inconnu')}**")
    else:
        st.warning("Veuillez entrer un texte.")
