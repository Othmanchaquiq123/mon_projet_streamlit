# Maintenance Predict

Pipeline complet (CPU ou GPU) pour  
* classifier le **type d’intervention**  
* prédire la **durée de travail** et la **durée d’arrêt**

## Installation

```bash
git clone https://…/mon_projet_maintenance.git
cd mon_projet_maintenance
pip install -r requirements.txt
python src/save_models.py      # entraîne et place les modèles dans /models
streamlit run app/app.py
