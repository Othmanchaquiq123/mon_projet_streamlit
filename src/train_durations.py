#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_durations.py – entraîne deux régressions :
• XGBoost  → DuréeTravail
• CatBoost → DuréeArrêt
Enregistre 4 pickles dans  <racine>/models/
"""

# ───────────────────────────────────────────────────────────
# 1) Imports & chemins
# ───────────────────────────────────────────────────────────
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from xgboost                   import XGBRegressor
from catboost                  import CatBoostRegressor
from utils_features            import add_calendar, remove_outliers_iqr   # <- ton fichier utilitaire

ROOT       = Path(__file__).resolve().parents[1]      # dossier projet
DATA_PATH  = ROOT / "data"   / "output.csv"
MODELS_DIR = ROOT / "models"                          # dossier commun
MODELS_DIR.mkdir(exist_ok=True)

# ───────────────────────────────────────────────────────────
# 2) Chargement / nettoyage
# ───────────────────────────────────────────────────────────
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV introuvable à {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
df["DateDebut"] = pd.to_datetime(df["DateDebut"], errors="coerce")
df.dropna(subset=["DateDebut","DureeTravail","DureeArret","Code_Machine"],
          inplace=True)

df = remove_outliers_iqr(df, "DureeTravail")
df = remove_outliers_iqr(df, "DureeArret")

df = add_calendar(df)
df["TimeIndex"] = (df.DateDebut - df.DateDebut.min()).dt.days
df = pd.get_dummies(df, columns=["Code_Machine"], prefix="", prefix_sep="")

# ───────────────────────────────────────────────────────────
# 3) Cibles & features
# ───────────────────────────────────────────────────────────
y_trav = np.log1p(df.DureeTravail)
y_stop = np.log1p(df.DureeArret)

ignored = ["DateDebut","DureeTravail","DureeArret",
           "Description_Dmd","Trv_Effectue"]
num_feats = (
    df.drop(columns=ignored, errors="ignore")
      .select_dtypes(include=[np.number])
      .columns
)

X = df[num_feats].copy()

# split chronologique 80 / 20
Xtr, Xte, ytr_tr, yte_tr = train_test_split(X, y_trav, test_size=.2, shuffle=False)
_,   _,  ytr_ar, yte_ar = train_test_split(X, y_stop, test_size=.2, shuffle=False)

# bool → uint8 puis scaling
bool_cols = Xtr.select_dtypes("bool").columns
Xtr[bool_cols] = Xtr[bool_cols].astype("uint8")
Xte[bool_cols] = Xte[bool_cols].astype("uint8")

to_scale = Xtr.columns.difference(bool_cols)
scaler = StandardScaler()
Xtr[to_scale] = scaler.fit_transform(Xtr[to_scale])
Xte[to_scale] = scaler.transform(Xte[to_scale])

# ───────────────────────────────────────────────────────────
# 4) Entraînement modèles
# ───────────────────────────────────────────────────────────
xgb = XGBRegressor(
    n_estimators=800, max_depth=6, learning_rate=0.06,
    subsample=.8, colsample_bytree=.8,
    objective="reg:squarederror", random_state=42
).fit(Xtr, ytr_tr)

cat = CatBoostRegressor(
    depth=8, learning_rate=0.05, iterations=1500,
    loss_function="RMSE", verbose=False, random_seed=42
).fit(Xtr, ytr_ar)

# ───────────────────────────────────────────────────────────
# 5) Sauvegarde pickles
# ───────────────────────────────────────────────────────────
joblib.dump(xgb,                MODELS_DIR / "xgb_tr.pkl")
joblib.dump(cat,                MODELS_DIR / "cat_ar.pkl")
joblib.dump(scaler,             MODELS_DIR / "scaler.pkl")
joblib.dump(num_feats.tolist(), MODELS_DIR / "num_feats.pkl")

print("✅  modèles durée enregistrés dans", MODELS_DIR)
