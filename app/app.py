# ──────────────────────────────────────────────────────────────
# app/app.py – Auth  ▸  Dashboard  ▸  Prédiction
# lance :  streamlit run app/app.py
# ──────────────────────────────────────────────────────────────
import sys, pathlib, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import joblib, scipy.sparse as sp

# =================  chemins + style  ==========================
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))                 # rendre src/ importable

css = pathlib.Path(__file__).with_name("style.css")
if css.exists():
    st.markdown(f"<style>{css.read_text()}</style>", unsafe_allow_html=True)

# =================  imports internes  =========================
from dashboard      import load_data, draw_dashboard      # app/dashboard.py
from utils_features import clean                          # src/utils_features.py

def _rerun():
    if hasattr(st, "rerun"): st.rerun()
    else: st.experimental_rerun()

# =================  modèles & data  ==========================
MODELS = ROOT / "models"
try:
    type_obj  = joblib.load(MODELS / "type_clf.pkl")   # dict{'model','tf_w','tf_c','id2name'/...}
    xgb_tr    = joblib.load(MODELS / "xgb_tr.pkl")
    cat_ar    = joblib.load(MODELS / "cat_ar.pkl")
    scaler    = joblib.load(MODELS / "scaler.pkl")
    num_feats = joblib.load(MODELS / "num_feats.pkl")  # list[str] (durées)
except FileNotFoundError as e:
    st.error(f"Modèle manquant : {e}. Lance d’abord `python -m src.save_models`.")
    st.stop()

# Libellés « humains » en priorité
ID2NAME       = type_obj.get("id2nice", type_obj.get("id2name", {}))
TYPE_NUM_COLS = type_obj.get("num_cols_type")    # list[str] si le classif type a des features num
TYPE_SCALER   = type_obj.get("scaler_type")      # scaler appliqué à ces features num

DATA_PATH = ROOT / "data" / "output.csv"
df        = load_data(DATA_PATH)
machines  = sorted(df.Code_Machine.unique())
MIN_DATE  = df.DateDebut.min()

# =================  priors par machine (CACHED)  ==============
@st.cache_data(show_spinner="Pré-calcul des probabilités par machine…", max_entries=1)
def build_machine_priors(df_: pd.DataFrame, _type_obj_: dict):
    """
    Retourne:
      - priors: dict {machine -> np.array([p(cat0), p(cat1), ...])}
      - classes: list (ordre des classes du modèle)
      - global_prior: np.array prior global (moyenne)
    NB: le 2e param commence par '_' pour éviter le hash du dict (Streamlit).
    """
    type_obj = _type_obj_
    model    = type_obj["model"]
    classes  = list(model.classes_) if hasattr(model, "classes_") else []

    # 1) Textes historiques nettoyés
    df_text = df_[["Code_Machine", "Trv_Effectue"]].dropna(subset=["Trv_Effectue"]).copy()
    if df_text.empty:
        return {}, classes, np.array([])

    txts = df_text["Trv_Effectue"].astype(str).map(clean).tolist()
    Xw   = type_obj["tf_w"].transform(txts)
    Xc   = type_obj["tf_c"].transform(txts)
    Xtxt = sp.hstack([Xw, Xc], format="csr")

    # 2) Si le modèle a été entraîné avec texte + NUM, ajoute un bloc NUM de même taille
    #    Ici on ne connaît pas les valeurs historiques ⇒ on met un bloc ZÉRO de dimension exacte.
    if TYPE_NUM_COLS is not None and len(TYPE_NUM_COLS) > 0:
        Xnum0 = sp.csr_matrix((Xtxt.shape[0], len(TYPE_NUM_COLS)), dtype=np.float32)
        X     = sp.hstack([Xtxt, Xnum0], format="csr")
    else:
        X = Xtxt

    # 3) Probas pour chaque ligne, puis agrégation par machine
    if hasattr(model, "predict_proba"):
        proba   = model.predict_proba(X)      # (n, C)
        classes = list(model.classes_)
        priors  = {}
        for m in df_text["Code_Machine"].unique():
            mask = (df_text["Code_Machine"] == m).values
            if mask.sum() == 0: 
                continue
            s = proba[mask].sum(axis=0) + 1.0             # lissage de Laplace
            priors[m] = (s / s.sum()).astype(float)
        global_prior = proba.mean(axis=0)
    else:
        pred    = model.predict(X)
        classes = list(np.unique(pred))
        priors  = {}
        for m in df_text["Code_Machine"].unique():
            mask = (df_text["Code_Machine"] == m).values
            bins = np.array([np.sum(pred[mask] == c) for c in classes], dtype=float) + 1.0
            priors[m] = bins / bins.sum()
        global_prior = np.ones(len(classes)) / len(classes)

    return priors, classes, global_prior

MACHINE_PRIORS, CLASSES, GLOBAL_PRIOR = build_machine_priors(df, type_obj)

# =================  Auth  ====================================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔐 Connexion")
    with st.form("login", clear_on_submit=True):
        pwd  = st.text_input("Mot de passe :", type="password")
        sub  = st.form_submit_button("Se connecter")
    if sub:
        if pwd.strip().lower() == "demo":   # 🔑 change ce mot de passe !
            st.session_state.auth_ok = True
            _rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()

if st.sidebar.button("🔒 Déconnexion"):
    st.session_state.auth_ok = False
    _rerun()
st.sidebar.success("🔓 authentifié")

# =================  Navigation  ==============================
page = st.sidebar.radio("Menu", ["Dashboard", "Prédiction"])

# ------------------ Dashboard --------------------------------
if page == "Dashboard":
    machine = st.selectbox("Choisir une machine :", machines, key="dash_machine")
    draw_dashboard(df, machine)
    st.stop()

# ------------------ Prédiction -------------------------------
st.header("Prédire une intervention")

c1, c2 = st.columns(2)
with c1:
    mach = st.selectbox("Machine", machines, key="p_m")
    jour = st.date_input("Date", value=dt.date.today(), key="p_d")
with c2:
    heure = st.time_input("Heure", value=dt.time(0, 0), key="p_h")
    texte = st.text_area("Description (facultatif)", key="p_t")

if st.button("Prédire"):
    # ===== (A) Type d’intervention : TEXTE (+ NUM si dispo) =====
    txt = clean(texte or "")
    Xw  = type_obj["tf_w"].transform([txt])
    Xc  = type_obj["tf_c"].transform([txt])
    parts = [sp.hstack([Xw, Xc], format="csr")]

    if TYPE_NUM_COLS is not None and TYPE_SCALER is not None and len(TYPE_NUM_COLS) > 0:
        dt_full = dt.datetime.combine(jour, heure)
        row_t = pd.Series(0.0, index=TYPE_NUM_COLS, dtype="float32")
        if mach in row_t.index: row_t[mach] = 1.0
        row_t["TimeIndex"] = (dt_full - MIN_DATE).days
        row_t["Year"], row_t["Month"], row_t["Day"] = dt_full.year, dt_full.month, dt_full.day
        row_t["Wday"] = dt_full.weekday()
        row_t["Month_sin"] = np.sin(2*np.pi*row_t["Month"]/12)
        row_t["Month_cos"] = np.cos(2*np.pi*row_t["Month"]/12)
        row_t["Wday_sin"]  = np.sin(2*np.pi*row_t["Wday"]/7)
        row_t["Wday_cos"]  = np.cos(2*np.pi*row_t["Wday"]/7)
        Xnum_t = pd.DataFrame([row_t])[TYPE_NUM_COLS]
        Xnum_t = pd.DataFrame(TYPE_SCALER.transform(Xnum_t), columns=TYPE_NUM_COLS)
        parts.append(sp.csr_matrix(Xnum_t.values))

    X_type = parts[0] if len(parts) == 1 else sp.hstack(parts, format="csr")
    model_type = type_obj["model"]

    # Proba texte
    classes = list(model_type.classes_) if hasattr(model_type, "classes_") else CLASSES
    if hasattr(model_type, "predict_proba"):
        proba_text = model_type.predict_proba(X_type)[0]
    else:
        # fallback one-hot si pas de predict_proba
        pred_id = int(model_type.predict(X_type)[0])
        proba_text = np.zeros(len(classes), dtype=float)
        try:
            proba_text[classes.index(pred_id)] = 1.0
        except Exception:
            proba_text[:] = 1.0 / len(classes)

    # Prior machine (même ordre de classes)
    prior_m = MACHINE_PRIORS.get(mach)
    if prior_m is None or len(prior_m) != len(classes):
        prior_m = GLOBAL_PRIOR if GLOBAL_PRIOR.size else np.ones(len(classes))/len(classes)

    # pondération : texte vs prior machine
    alpha = 0.75 if len(txt.split()) >= 3 else 0.35
    proba_mix = alpha * proba_text + (1 - alpha) * prior_m
    proba_mix = proba_mix / proba_mix.sum()

    top3 = np.argsort(proba_mix)[-3:][::-1]
    lines = []
    for i in top3:
        cid = int(classes[i])
        nice = ID2NAME.get(cid, f"cat_{cid}")
        lines.append(f"• {nice} — {proba_mix[i]*100:.1f}%")

    best_cid = int(classes[top3[0]])
    label    = ID2NAME.get(best_cid, f"cat_{best_cid}")
    conf_tag = " *(faible confiance)*" if proba_mix[top3[0]] < 0.45 else ""

    # ===== (B) Durées Travail / Arrêt =========================
    dt_full = dt.datetime.combine(jour, heure)
    row = pd.Series(0.0, index=num_feats, dtype="float32")
    if mach in row.index: row[mach] = 1.0
    row["TimeIndex"] = (dt_full - MIN_DATE).days
    row["Year"], row["Month"], row["Day"] = dt_full.year, dt_full.month, dt_full.day
    row["Wday"] = dt_full.weekday()
    row["Month_sin"] = np.sin(2*np.pi*row["Month"]/12)
    row["Month_cos"] = np.cos(2*np.pi*row["Month"]/12)
    row["Wday_sin"]  = np.sin(2*np.pi*row["Wday"]/7)
    row["Wday_cos"]  = np.cos(2*np.pi*row["Wday"]/7)

    Xnum = pd.DataFrame([row])[num_feats]
    bc   = Xnum.select_dtypes("bool").columns
    Xnum[bc] = Xnum[bc].astype("uint8")
    Xnum[num_feats] = scaler.transform(Xnum[num_feats])

    dur_tr = float(np.expm1(xgb_tr.predict(Xnum))[0])
    dur_ar = float(np.expm1(cat_ar.predict(Xnum))[0])

    # ===== Affichage ==========================================
    st.success(f"**Type prédit :** {label}{conf_tag}")
    st.info(f"**Durée travail :** {dur_tr:.1f} h   **Durée arrêt :** {dur_ar:.1f} h")

    with st.expander("Pourquoi ce type ?"):
        st.markdown("\n".join(lines))
        if len((texte or "").strip()) == 0:
            st.caption("Texte vide — prior machine utilisé.")
