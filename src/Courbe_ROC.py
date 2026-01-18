import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import streamlit as st

def afficher_courbe_roc(df_test: pd.DataFrame, model_path: str = "models/xgboost_model.joblib"):
    if "Non_renouvellement" not in df_test.columns:
        st.warning("❌ La colonne 'Non_renouvellement' est absente du jeu de données.")
        return

    # 🔀 Séparer X et y (ne pas supprimer 'flag_actif')
    X = df_test.drop(columns=["Non_renouvellement", "No du Contrat"], errors="ignore")
    y = df_test["Non_renouvellement"]

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Modèle introuvable : {model_path}")
        return

    try:
        # 🔮 Prédire les probabilités
        y_proba = model.predict_proba(X)[:, 1]
    except ValueError as e:
        st.error(f"❌ Erreur de prédiction : {e}")
        return

    # 📈 Calcul de la courbe ROC
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    # 📊 Création du graphique
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", label="Aléatoire", alpha=0.6)
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR)")
    ax.set_title("Courbe ROC - Modèle XGBoost")
    ax.legend(loc="lower right")
    ax.grid(True)

    # ✅ Affichage centré et réduit
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig)

    # 🔎 Vue élargie en option
    with st.expander("🔍 Agrandir la courbe ROC"):
        st.pyplot(fig)