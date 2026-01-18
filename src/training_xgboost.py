import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

def entrainer_xgboost(df_model: pd.DataFrame):
    st.info("🚀 Entraînement du modèle XGBoost...")

    # 🔀 Séparation des variables explicatives (X) et de la cible (y)
    X = df_model.drop(columns=["No du Contrat", "Non_renouvellement"], errors="ignore")
    y = df_model["Non_renouvellement"]

    # ✂ Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ⚙ Initialisation du modèle
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # 🧠 Entraînement
    model.fit(X_train, y_train)

    # 💾 Sauvegarde du modèle avec les noms des features
    os.makedirs("models", exist_ok=True)
    booster = model.get_booster()
    booster.feature_names = list(X.columns)
    joblib.dump(model, "models/xgboost_model.joblib")

    # 📊 Matrice de confusion
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # 📈 Affichage de la matrice
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_title("Matrice de confusion - XGBoost")
    ax_cm.set_xlabel("Label prédit")
    ax_cm.set_ylabel("Label réel")
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/confusion_matrix_xgboost.png")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig_cm)

    # ⬇ Bouton de téléchargement
    with open("outputs/figures/confusion_matrix_xgboost.png", "rb") as f:
        st.download_button(
            label="📥 Télécharger la matrice de confusion",
            data=f,
            file_name="confusion_matrix_xgboost.png",
            mime="image/png"
        )

    # 🔍 Résumé métier
    total = cm.sum()
    bonnes_predictions = cm[0][0] + cm[1][1]
    erreurs = cm[0][1] + cm[1][0]
    taux_bonnes_pred = bonnes_predictions / total
    taux_erreurs = erreurs / total

    st.markdown("### 🧾 Résumé des résultats")
    st.markdown(
        f"""
        - ✅ *Bonnes prédictions* : {bonnes_predictions} contrats correctement identifiés comme renouvelés ou non renouvelés.
        - ❌ *Erreurs de prédiction* : {erreurs} contrats mal prédits.
        """
    )

    # 📘 Explication finale pour les équipes métier
    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:16px; line-height:1.6;">
        ℹ <strong>Sur un total de <u>{total}</u> contrats analysés</strong>, 
        le modèle XGBoost a correctement prédit <strong>{bonnes_predictions}</strong> d'entre eux, 
        soit un taux de bonne prédiction de <strong>{taux_bonnes_pred:.1%}</strong>.
        <br>Il s'est trompé sur <strong>{erreurs}</strong> contrats.
        <br><br>✅ Cela montre que le modèle est globalement performant, 
        tout en laissant place à de futures améliorations.
        </div>
        """,
        unsafe_allow_html=True
    )

    # 📊 Graphique circulaire
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie(
        [bonnes_predictions, erreurs],
        labels=["Bonnes prédictions", "Erreurs"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=90,
        wedgeprops=dict(width=0.5)
    )
    ax_pie.axis("equal")

    # 🎯 Titre et espacement centré
    st.markdown("<div style='margin-top: 30px; text-align: center;'><h4>📊 Taux global de bonne prédiction</h4></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig_pie)

    # ✅ Confirmation finale
    st.success("✅ Modèle XGBoost entraîné avec succès")