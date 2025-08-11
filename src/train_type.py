# src/train_type.py
from pathlib import Path
import json, re
import joblib, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from utils_features import clean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "output.csv"
OUT  = ROOT / "models"; OUT.mkdir(exist_ok=True)
LABELS_JSON = ROOT / "config" / "type_labels.json"

# ------------------------------------------------------------
# 0) Data + features
# ------------------------------------------------------------
df = pd.read_csv(DATA, sep=";", encoding="latin1")
df["txt"] = df["Trv_Effectue"].fillna("").map(clean)
df["DateDebut"] = pd.to_datetime(df["DateDebut"], errors="coerce")
df = df.dropna(subset=["DateDebut", "Code_Machine"])

# calendrier
df["TimeIndex"] = (df.DateDebut - df.DateDebut.min()).dt.days
df["Year"]  = df.DateDebut.dt.year
df["Month"] = df.DateDebut.dt.month
df["Day"]   = df.DateDebut.dt.day
df["Wday"]  = df.DateDebut.dt.weekday
df["Month_sin"] = np.sin(2*np.pi*df["Month"]/12)
df["Month_cos"] = np.cos(2*np.pi*df["Month"]/12)
df["Wday_sin"]  = np.sin(2*np.pi*df["Wday"]/7)
df["Wday_cos"]  = np.cos(2*np.pi*df["Wday"]/7)

# one-hot machine
df = pd.get_dummies(df, columns=["Code_Machine"], prefix="", prefix_sep="")

# TF-IDF
tf_w = TfidfVectorizer(max_df=.8, min_df=3, ngram_range=(1,2), sublinear_tf=True)
tf_c = TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=30000, sublinear_tf=True)
Xw = tf_w.fit_transform(df["txt"])
Xc = tf_c.fit_transform(df["txt"])

# labels (existants sinon KMeans)
if "Cat" in df.columns:
    y = df["Cat"].astype(int).values
else:
    km = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=2048)
    y = km.fit_predict(Xw)

# numériques utilisés par le classif type
ignored = {"DateDebut","txt","Trv_Effectue","DureeTravail","DureeArret","Cat","Cat_name"}
num_cols_type = [c for c in df.columns if c not in ignored and pd.api.types.is_numeric_dtype(df[c])]
scaler_type = StandardScaler()
Xnum_type = scaler_type.fit_transform(df[num_cols_type].astype("float32"))

# concat global pour classif type
X = sp.hstack([Xw, Xc, Xnum_type], format="csr")

# split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

# modèle (log-loss pour avoir les proba)
clf = SGDClassifier(loss="log_loss", max_iter=6000, class_weight="balanced", random_state=42)
clf.fit(Xtr, ytr)

# ------------------------------------------------------------
# 1) Libellés lisibles
# ------------------------------------------------------------
# a) libellés « bruts » par top mots (fallback)
terms_w = np.array(tf_w.get_feature_names_out())
id2name = {}
for cid in np.unique(y):
    mask = (y == cid)
    if mask.any():
        mean_tf = Xw[mask].mean(axis=0).A1
        top = terms_w[mean_tf.argsort()[::-1][:4]]
        id2name[int(cid)] = "_".join(top) if len(top) else f"cat_{int(cid)}"
    else:
        id2name[int(cid)] = f"cat_{int(cid)}"

# b) canonisation via table de correspondance
rules = {}
if LABELS_JSON.exists():
    rules = {re.compile(pat): nice for pat, nice in json.loads(LABELS_JSON.read_text()).items()}
else:
    # mini-fallback si le JSON n’existe pas
    rules = {
        re.compile(r"(?i)marche|mise"): "Mise en marche",
        re.compile(r"(?i)reglage|capteur|position"): "Réglage position / capteur",
        re.compile(r"(?i)nettoyage|verification|deblocage|reparation"): "Nettoyage / vérification / déblocage / réparation",
        re.compile(r"(?i)changement|roulement|poulie"): "Changement roulement / poulie",
        re.compile(r"(?i)bobinoir|bobinoire|debouchage"): "Déblocage bobinoir / débouchage",
        re.compile(r"(?i)serrage|depart|pointe"): "Capteur / serrage / départ pointe",
    }

def canonize(label_raw: str) -> str:
    # on teste tous les motifs sur le label brut
    for pat, nice in rules.items():
        if pat.search(label_raw.replace("_", " ")):
            return nice
    return label_raw  # si rien ne matche, on garde le brut

id2nice = {cid: canonize(lbl) for cid, lbl in id2name.items()}

# c) priors par machine (utile texte vide)
priors = {}
mach_cols = [c for c in num_cols_type if c.startswith("M")]
if mach_cols and "Cat" in df.columns:
    for m in mach_cols:
        sub = df[df[m] == 1]
        if len(sub):
            p = sub["Cat"].value_counts(normalize=True).to_dict()
            priors[m] = {int(k): float(v) for k, v in p.items()}

# ------------------------------------------------------------
# 2) Sauvegarde
# ------------------------------------------------------------
joblib.dump({
    "model": clf,
    "tf_w": tf_w,
    "tf_c": tf_c,
    "num_cols_type": num_cols_type,
    "scaler_type": scaler_type,
    "id2name": id2name,   # bruts
    "id2nice": id2nice,   # lisibles
    "priors": priors
}, OUT / "type_clf.pkl")

print("✅ type_clf.pkl (texte + machine + calendrier) avec libellés lisibles enregistré.")
