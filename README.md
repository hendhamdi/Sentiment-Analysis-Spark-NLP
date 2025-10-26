# 🧠 Student Sentiment Analysis - Spark NLP

This project performs **sentiment analysis** on reviews from MP2L Master’s students using a text-processing pipeline based on **Apache Spark (PySpark)** and a **logistic regression model**.

---

## 🚀 Objectives

- Automate the **classification of student reviews** (positive, neutral, negative)
- Visualize the distribution of sentiments globally, by semester, and by year
- Provide an **interactive web interface** to dynamically explore the results


## 🔧 Technologies Used

- **Python 3**:  Main programming language
- **Apache Spark (PySpark)**: Distributed computing framework for large-scale data processing
- **Flask**: Web interface
- **Spark MLlib** : Machine learning library in Spark used for logistic regression
- **Hadoop** (requires `winutils.exe` configuration on Windows)
- **Bibliothèques Python** :
  - `pyspark`
  - `pandas`
  - `matplotlib`
  - `PrettyTable`|
  - `json`
  - `pathlib`
|

## 📂 Project Structure

```plaintext
sentiment-analysis-spark/
├── data/
│ └── avis_etudiants_dataset.csv    # Student reviews dataset
├── src/
│ ├── main.py                       # Main analysis script
│ └── webapp/                       # Flask interface
│ ├── app.py
│ ├── static/
│ │ └── style.css
│ ├── templates/
│ │ └── index.html
│ └── images/
│ ├── uvt_logo.png
│ └── isi_logo.png
├── output/                          # Results
│ ├── results.txt                    # Detailed predictions for 15 samples
│ ├── results.png                    # Sentiment distribution chart
│ ├── sentiments_par_annee.json      # Data by year (for the web interface)
│ └── sentiments_par_semestre.json   # Data by semester (for the web interface)
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

- Précision du modèle : ~86%
- Prédictions (extrait) sauvegardées dans output/results.txt.
- Graphique global des sentiments généré automatiquement :

![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis---Spark-NLP/blob/main/output/results.png)

- Graphiques interactifs par année disponibles dans l'interface Flask (chargés depuis sentiments_par_annee.json)
- Graphiques interactifs par semestre disponibles dans l'interface Flask (chargés depuis sentiments_par_semestre.json)


## 📈 Fonctionnalités de l'interface web

- Visualisation globale de la répartition des sentiments
- Graphiques interactifs par année et semestre
- Affichage dynamique avec Chart.js
- Navigation fluide et design épuré
- Interface responsive et épurée
![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis-Spark-NLP/blob/main/src/webapp/images/Interface-Web1.png)
![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis-Spark-NLP/blob/main/src/webapp/images/Interface-Web2.png)


## 🚀 Améliorations futures

- Ajout de nouvelles sources de données (questionnaires, forums, etc.)
- Génération automatique de recommandations pédagogiques
- Système d'authentification pour un accès personnalisé

