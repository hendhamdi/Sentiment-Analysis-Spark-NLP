# 🧠 Analyse de Sentiments Étudiants - Spark NLP

Ce projet réalise une **analyse de sentiments** sur des avis d'étudiants du Master MP2L, à l'aide d'un pipeline de traitement de texte basé sur **Apache Spark (PySpark)** et un modèle de **régression logistique**.

---

## 🚀 Objectifs

- Automatiser la **classification des avis étudiants** (positif, neutre, négatif)
- Visualiser la répartition des sentiments de manière globale, par semestre et par année
- Offrir une **interface web interactive** pour explorer les résultats dynamiquement


## 🔧 Technologies utilisées

- **Python 3**:  Langage principal du projet.
- **Apache Spark (PySpark)**: Framework de calcul distribué pour le traitement des données massives.
- **Flask** - Interface web
- **Spark MLlib** : Librairie de machine learning incluse dans Spark pour la régression logistique.
- **Hadoop** (configuration de `winutils.exe` nécessaire sous Windows)
- **Bibliothèques Python** :
  - `pyspark`
  - `pandas`
  - `matplotlib`
  - `PrettyTable`|
  - `json`
  - `pathlib`
|

## 📂 Structure du Projet

```plaintext
sentiment-analysis-spark/
├── data/
│ └── avis_etudiants_dataset.csv    # Dataset des avis
├── src/
│ ├── main.py                       # Script principal d'analyse
│ └── webapp/                       # Interface Flask
│ ├── app.py
│ ├── static/
│ │ └── style.css
│ ├── templates/
│ │ └── index.html
│ └── images/
│ ├── uvt_logo.png
│ └── isi_logo.png
├── output/                          # Résultats
│ ├── results.txt                    # Prédictions détaillées de 15 exemples
│ ├── results.png                    # Graphique des sentiments
│ ├── sentiments_par_annee.json      # Données par année (pour l'interface web)
│ └── sentiments_par_semestre.json   # Données par semestre (pour l'interface web)
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


## 🚀 Améliorations futures

- Ajout de nouvelles sources de données (questionnaires, forums, etc.)
- Génération automatique de recommandations pédagogiques
- Système d'authentification pour un accès personnalisé

