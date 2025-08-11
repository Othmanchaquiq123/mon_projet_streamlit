"""
utils_features.py – fonctions réutilisables
"""
import re, unicodedata, numpy as np, pandas as pd

# ─────────────────────────────────────────────
# NETTOYAGE TEXTE
# ─────────────────────────────────────────────
def clean(txt: str) -> str:
    """Nettoie un texte (lower, accents, ponctuation)."""
    if pd.isna(txt):
        return ""
    txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^a-zA-Z\s]", " ", txt.lower())
    return re.sub(r"\s+", " ", txt).strip()

# ─────────────────────────────────────────────
# FEATURE ENGINEERING COMMUN
# ─────────────────────────────────────────────
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"]  = df.DateDebut.dt.year
    df["Month"] = df.DateDebut.dt.month
    df["Day"]   = df.DateDebut.dt.day
    df["Wday"]  = df.DateDebut.dt.weekday
    df["Month_sin"] = np.sin(2*np.pi*df.Month/12)
    df["Month_cos"] = np.cos(2*np.pi*df.Month/12)
    df["Wday_sin"]  = np.sin(2*np.pi*df.Wday/7)
    df["Wday_cos"]  = np.cos(2*np.pi*df.Wday/7)
    return df

def remove_outliers_iqr(df: pd.DataFrame, col: str, k: float = 1.5):
    q1, q3 = df[col].quantile([.25, .75])
    iqr = q3 - q1
    return df[(df[col] >= q1 - k*iqr) & (df[col] <= q3 + k*iqr)]
