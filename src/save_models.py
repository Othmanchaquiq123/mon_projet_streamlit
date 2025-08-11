# ─── save_models.py (corrigé) ──────────────────────────────────────────
import pathlib, importlib, sys, joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]   # dossier projet
sys.path.insert(0, str(ROOT))                       # <- ★ on ajoute le projet à PYTHONPATH

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

required = [
    MODELS_DIR/"type_clf.pkl",
    MODELS_DIR/"xgb_tr.pkl",
    MODELS_DIR/"cat_ar.pkl",
    MODELS_DIR/"scaler.pkl",
    MODELS_DIR/"num_feats.pkl",
]

missing = [p for p in required if not p.exists()]
if not missing:
    print("✅  Tous les modèles déjà présents – rien à faire.")
    sys.exit(0)

print("⚠️  Modèles manquants → entraînement…")

# import dynamique = exécution des scripts
importlib.import_module("src.train_type")      # fonctionne une fois src/ est un package
importlib.import_module("src.train_durations")

# contrôle
for p in required:
    if not p.exists():
        raise FileNotFoundError(f"{p} absent après entraînement !")
print("✅  Modèles créés dans /models")
