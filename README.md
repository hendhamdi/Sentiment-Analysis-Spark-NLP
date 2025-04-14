# 🧠 Analyse des Sentiments - Spark NLP

Ce projet consiste à effectuer une **analyse de sentiments** sur des avis utilisateurs en utilisant un pipeline **Spark (PySpark)** combiné à un modèle de **régression logistique**.  
Chaque avis est automatiquement classé comme **positif**, **neutre** ou **négatif** en fonction de son contenu textuel.  
Une visualisation des résultats est également générée, illustrant la répartition des sentiments à l'aide d’un graphique.

---


## 🔧 Technologies utilisées

- **Python 3**
- **Apache Spark (PySpark)**
- **Hadoop** (configuration de `winutils.exe` nécessaire sous Windows)
- **Bibliothèques Python** :
  - `pyspark`
  - `pandas`
  - `matplotlib`
- **MLlib** (pour la régression logistique)


---


## 📊 Résultats

- Précision du modèle : 85 %
- Visualisation des résultats sous forme de graphique (voir `results.png`) montrant la répartition des prédictions :
![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis---Spark-NLP/blob/main/output/results.png)
