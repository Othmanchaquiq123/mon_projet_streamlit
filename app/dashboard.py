# app/dashboard.py
"""
Dashboard Streamlit : graphiques + commentaires
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set_theme(style="whitegrid")

# ─────────────────────────────────────────────────────────────
# Chargement dataset + nettoyage minimum
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin1")
    df["DateDebut"] = pd.to_datetime(df["DateDebut"], errors="coerce")
    # durées numériques
    df["DureeTravail"] = pd.to_numeric(df["DureeTravail"], errors="coerce")
    df["DureeArret"]   = pd.to_numeric(df["DureeArret"],   errors="coerce")
    # lignes valides uniquement
    df = df.dropna(subset=["DateDebut", "Code_Machine", "DureeTravail", "DureeArret"])
    # clip léger pour éviter valeurs négatives
    df["DureeTravail"] = df["DureeTravail"].clip(lower=0)
    df["DureeArret"]   = df["DureeArret"].clip(lower=0)
    return df

# ─────────────────────────────────────────────────────────────
# Commentaire auto pour une machine
# ─────────────────────────────────────────────────────────────
def _comment_block(code: str, s_tr: pd.Series, s_ar: pd.Series) -> str:
    q_tr = s_tr.quantile([.25, .5, .75])
    q_ar = s_ar.quantile([.25, .5, .75])
    return f"""
### 🔧 Machine {code}

**Travail** — médiane **{q_tr[.5]:.1f} h**, IQR **{(q_tr[.75]-q_tr[.25]):.1f} h** · min **{s_tr.min():.1f} h** · max **{s_tr.max():.1f} h**  
**Arrêt** — médiane **{q_ar[.5]:.1f} h**, IQR **{(q_ar[.75]-q_ar[.25]):.1f} h** · min **{s_ar.min():.1f} h** · max **{s_ar.max():.1f} h**

**Analyse rapide :**
- Interventions **courtes** (travail \< {q_tr[.25]:.1f} h) = routines ; **longues** (travail \> {q_tr[.75]:.1f} h) à **auditer**.
- Si **arrêts fréquents/longs** (arrêt \> {q_ar[.75]:.1f} h), vérifier pièces d’usure, réglages et planning.
"""

# ─────────────────────────────────────────────────────────────
# Dessin du dashboard pour une machine
# ─────────────────────────────────────────────────────────────
def draw_dashboard(df: pd.DataFrame, machine: str):
    sub = df[df["Code_Machine"] == machine].copy()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Interventions", len(sub))
    with c2: st.metric("Travail moyen (h)", f"{sub.DureeTravail.mean():.2f}")
    with c3: st.metric("Arrêt moyen (h)",   f"{sub.DureeArret.mean():.2f}")
    # st.metric n’accepte pas datetime → string courte pour éviter la coupure
    with c4: st.metric("Dernière date",     sub.DateDebut.max().strftime("%d/%m/%Y"))

    # ── Agrégation journalière (resample ‘D’) ─────────────────────
    daily = (
        sub.set_index("DateDebut")[["DureeTravail", "DureeArret"]]
           .resample("D").sum()
           .fillna(0.0)
    )
    daily_ma7 = daily.rolling(7, min_periods=1).mean()

    st.subheader("Durées agrégées par jour (somme) + moyenne mobile 7j")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(daily.index,     daily["DureeTravail"], label="Travail (jour)")
    ax.plot(daily.index,     daily["DureeArret"],   label="Arrêt (jour)")
    ax.plot(daily_ma7.index, daily_ma7["DureeTravail"], linestyle="--", label="Travail • MM7")
    ax.plot(daily_ma7.index, daily_ma7["DureeArret"],   linestyle="--", label="Arrêt • MM7")
    ax.set_xlabel("Date"); ax.set_ylabel("Heures")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(); ax.grid(True); st.pyplot(fig)

    # ── Histogrammes + KDE pour les deux durées ───────────────────
    st.subheader("Distributions (histogrammes)")
    ch1, ch2 = st.columns(2)
    with ch1:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(sub, x="DureeTravail", bins=30, kde=True, ax=ax)
        ax.set_xlabel("Durée travail (h)"); st.pyplot(fig)
    with ch2:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.histplot(sub, x="DureeArret", bins=30, kde=True, ax=ax)
        ax.set_xlabel("Durée arrêt (h)"); st.pyplot(fig)

    # ── Violinplots pour les deux durées ──────────────────────────
    st.subheader("Violinplots (répartition fine)")
    v1, v2 = st.columns(2)
    with v1:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.violinplot(y=sub["DureeTravail"], inner="quartile", ax=ax)
        ax.set_ylabel("Durée travail (h)"); ax.set_xlabel(""); st.pyplot(fig)
    with v2:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.violinplot(y=sub["DureeArret"], inner="quartile", ax=ax)
        ax.set_ylabel("Durée arrêt (h)"); ax.set_xlabel(""); st.pyplot(fig)

    # ── Boxplots pour les deux durées (ajout demandé) ─────────────
    st.subheader("Boxplots")
    b1, b2 = st.columns(2)
    with b1:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(y=sub["DureeTravail"], ax=ax, whis=1.5)  # outliers visibles
        ax.set_ylabel("Durée travail (h)"); ax.set_xlabel(""); st.pyplot(fig)
    with b2:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(y=sub["DureeArret"], ax=ax, whis=1.5)
        ax.set_ylabel("Durée arrêt (h)"); ax.set_xlabel(""); st.pyplot(fig)

    # ── Scatter : Travail vs Arrêt ────────────────────────────────
    st.subheader("Nuage de points : Travail vs Arrêt")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=sub, x="DureeTravail", y="DureeArret",
                    edgecolor="k", linewidth=0.3, ax=ax, alpha=.85)
    ax.set_xlabel("Durée travail (h)"); ax.set_ylabel("Durée arrêt (h)")
    ax.grid(True); st.pyplot(fig)

    # ── Commentaires auto ─────────────────────────────────────────
    st.markdown(_comment_block(machine, sub["DureeTravail"], sub["DureeArret"]))
