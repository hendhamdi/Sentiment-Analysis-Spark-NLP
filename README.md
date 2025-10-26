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

## 🚀 How to Run the Project

1. **Run the data analysis**:

```bash
python src/main.py
```
2. **Launch the web interface**:

```bash
python src/webapp/app.py
```

## 📊 Results

- Model accuracy: ~86%
- Predictions (sample) saved in output/results.txt
- Global sentiment distribution chart automatically generated:

![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis---Spark-NLP/blob/main/output/results.png)

- Interactive charts by year available in the Flask interface (loaded from sentiments_by_year.json)
- Interactive charts by semester available in the Flask interface (loaded from sentiments_par_semestre.json)


## 📈 Web Interface Features

- Global sentiment distribution visualization
- Interactive charts by year and semester
- Dynamic display using Chart.js
- Smooth navigation and clean design
- Responsive and minimalist interface
![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis-Spark-NLP/blob/main/src/webapp/images/Interface-Web1.png)
![Répartition des sentiments](https://github.com/hendhamdi/Sentiment-Analysis-Spark-NLP/blob/main/src/webapp/images/Interface-Web2.png)


## 🚀 Future Improvements

- Add new data sources (surveys, forums, etc.)
- Automatically generate educational recommendations
- Implement an authentication system for personalized access

