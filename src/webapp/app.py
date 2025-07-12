# Importation des modules 
from flask import Flask, render_template, send_from_directory, jsonify
import os
import sys
import json
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Création de l'application Flask
app = Flask(__name__)

# Définition des chemins vers les fichiers générés par l'analyse
OUTPUT_FOLDER = project_root / 'output'
RESULTS_FILE = OUTPUT_FOLDER / 'results.txt'
GRAPH_FILE = 'results.png'

# Affichage des informations utiles pour le débogage
print("\n=== Debug Information ===")
print("Current directory:", current_dir)
print("Project root:", project_root)
print("Output folder:", OUTPUT_FOLDER)
print("Semestre file exists:", (OUTPUT_FOLDER / "sentiments_par_semestre.json").exists())
print("=========================\n")

# Fonction pour charger les résultats d’analyse depuis `main.py`
def load_analysis_results():
    """Charge les résultats depuis main.py"""
    try:
        from main import get_analysis_results
        return get_analysis_results()
    except ImportError as e:
        print(f"Erreur d'import: {e}")
        return None

# Route page d’accueil
@app.route("/")
def index():
    results = load_analysis_results() # Tente de charger les résultats dynamiquement
    
    if results is None:
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                table_data = f.read()
        except:
            table_data = "Aucun résultat disponible"
        
        return render_template("index.html", 
                            predictions=table_data,
                            accuracy="N/A",
                            examples=[])

    return render_template("index.html",
                         predictions=results['table'],
                         accuracy=f"{results['accuracy']:.2f}",
                         examples=results['examples'])
# Route pour servir le graphique généré (image PNG)
@app.route("/graph")
def show_graph():
    return send_from_directory(OUTPUT_FOLDER, GRAPH_FILE)

# Route pour retourner les sentiments par année au format JSON
@app.route("/data/sentiments")
def sentiments_par_annee():
    try:
        with open(OUTPUT_FOLDER / "sentiments_par_annee.json", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route pour retourner les sentiments par semestre au format JSON
@app.route("/data/sentiments_semestre")
def sentiments_par_semestre():
    semestre_file = OUTPUT_FOLDER / "sentiments_par_semestre.json"
        # Si le fichier n’existe pas, retourne une erreur 404

    if not semestre_file.exists():
        return jsonify({
            "error": "File not found",
            "path": str(semestre_file),
            "available_files": [f.name for f in OUTPUT_FOLDER.glob("*") if f.is_file()]
        }), 404
    
    try:
        with open(semestre_file, encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError as e:
        return jsonify({"error": "Invalid JSON", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

# Route pour servir des images statiques depuis un dossier `images`
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'images'), filename)

# Lancement de l'application en mode debug sur le port 5000
if __name__ == "__main__":
    app.run(debug=True, port=5000)