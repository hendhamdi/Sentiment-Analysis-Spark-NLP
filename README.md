# 🧠 Analyse des Sentiments - Spark NLP

Ce projet consiste à effectuer une **analyse de sentiments** sur des avis utilisateurs en utilisant un pipeline **Spark (PySpark)** combiné à un modèle de **régression logistique**.  
Chaque avis est automatiquement classé comme **positif**, **neutre** ou **négatif** en fonction de son contenu textuel.  
Une visualisation des résultats est également générée, illustrant la répartition des sentiments à l'aide d’un graphique.



## 🔧 Technologies utilisées

- **Python 3**:  Langage principal du projet.
- **Apache Spark (PySpark)**: Framework de calcul distribué pour le traitement des données massives.
- **Spark MLlib** : Librairie de machine learning incluse dans Spark pour la régression logistique.
- **Hadoop** (configuration de `winutils.exe` nécessaire sous Windows)
- **Bibliothèques Python** :
  - `pyspark`
  - `pandas`
  - `matplotlib`


## 📂 Structure du Projet

```plaintext
sentiment-analysis-spark/
├── data/                  # Données brutes
|   └── Reviews.csv        # Dataset
├── src/                   # Scripts principaux
│   └── main.py            # Script d'analyse des sentiments
├── output/                # Résultats générés
│   ├── results.txt        # tableau de prédictions
│   └── results.png        # Graphique de répartition des sentiments
└── README.md    
 ``` 



## 📊 Résultats

- Le modèle entraîné atteint 85% de précision sur les données de test.
- Visualisation des résultats sous forme de tableau (voir `results.text`) montrant des exemples de prédictions.
- Visualisation des résultats sous forme de graphique (voir `results.png`) montrant la répartition des prédictions :

![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis---Spark-NLP/blob/main/output/results.png)

## 🚀 Améliorations futures
Intégrer une interface utilisateur web permettant aux utilisateurs de :

- Visualiser dynamiquement les résultats sous forme de graphiques interactifs.
- Saisir un avis directement depuis une page web et obtenir instantanément la prédiction du sentiment (positif, neutre ou négatif).