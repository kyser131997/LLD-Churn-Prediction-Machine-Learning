import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def comparer_modeles_streamlit(df_modele):
    """
    Compare les performances de 3 modèles de classification et affiche les résultats dans Streamlit.
    
    Args:
        df_modele (pd.DataFrame): Données prétraitées avec les colonnes 'Non_renouvellement' et 'No du Contrat'.
    """

    #st.subheader("📊 Comparaison des modèles de prédiction")

    # 🔀 Séparer X et y
    X = df_modele.drop(columns=["Non_renouvellement", "No du Contrat"], errors="ignore")
    y = df_modele["Non_renouvellement"]

    # ✂ Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔁 Liste des modèles à comparer
    modeles = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            solver="liblinear", class_weight="balanced", random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=5, scale_pos_weight=1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
    }

    resultats = []

    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        resultats.append({
            "Modèle": nom,
            "F1-score": round(f1, 4),
            "Recall (classe 1)": round(recall, 4),
            "Précision": round(precision, 4)
        })

    # 🔄 Créer DataFrame résultats
    df_resultats = pd.DataFrame(resultats)
    st.dataframe(df_resultats)

    # 🔄 Réorganiser pour graphique
    df_melted = df_resultats.melt(id_vars="Modèle", var_name="Métrique", value_name="Score")

    # 🎨 Graphe Streamlit
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_melted, x="Modèle", y="Score", hue="Métrique", ax=ax)
    plt.ylim(0.5, 0.9)
    plt.title("Comparaison dynamique des modèles")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.legend(title="Métrique")
    st.pyplot(fig)