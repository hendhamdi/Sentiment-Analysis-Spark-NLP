# 🧠 Analyse des Sentiments - Spark NLP


Ce projet effectue une **analyse de sentiments** sur les avis des étudiants du Master MP2L en utilisant **Apache Spark (PySpark)** avec un pipeline complet de NLP et un modèle de **régression logistique**.


## 📌 Fonctionnalités

- Classification automatique des avis en **positif**, **neutre** ou **négatif**
- Génération de visualisations claires
- Interface web intégrée pour explorer les résultats


## 🔧 Technologies utilisées

- **Python 3**:  Langage principal du projet.
- **Apache Spark (PySpark)**: Framework de calcul distribué pour le traitement des données massives.
- **Spark MLlib** : Librairie de machine learning incluse dans Spark pour la régression logistique.
- **Hadoop** (configuration de `winutils.exe` nécessaire sous Windows)
- **Flask** - Interface web
- **Bibliothèques Python** :
  - `pyspark`
  - `pandas`
  - `matplotlib`
  - `PrettyTable`|


## 📂 Structure du Projet

```plaintext
ssentiment-analysis-spark/
├── data/
│ └── avis_etudiants_dataset.csv # Dataset des avis
├── src/
│ ├── main.py # Script principal d'analyse
│ └── webapp/ # Interface Flask
│ ├── app.py
│ ├── static/
│ │ └── style.css
│ ├── templates/
│ │ └── index.html
│ └── images/
│ ├── uvt_logo.png
│ └── isi_logo.png
├── output/ # Résultats
│ ├── results.txt # Prédictions détaillées
│ └── results.png # Graphique des sentiments
└── README.md
 ``` 

## 🚀 Comment lancer le projet

1. **Analyse des données**:

```bash
python src/main.py
```
2. **Interface web**:

```bash
python src/webapp/app.py
```

## 📊 Résultats

- Le modèle entraîné atteint 85% de précision sur les données de test.
- Visualisation des résultats sous forme de tableau (voir `results.text`) montrant des exemples de prédictions.
- Visualisation des résultats sous forme de graphique (voir `results.png`) montrant la répartition des prédictions :

![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis---Spark-NLP/blob/main/output/results.png)

## 🚀 Améliorations futures

Intégrer une interface utilisateur web permettant aux utilisateurs de :

- Visualiser dynamiquement les résultats sous forme de graphiques interactifs.
- Saisir un avis directement depuis une page web et obtenir instantanément la prédiction du sentiment (positif, neutre ou négatif).import logging

