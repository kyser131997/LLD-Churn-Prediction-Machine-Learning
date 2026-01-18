import streamlit as st
import pandas as pd
import os
import io
from PIL import Image

from src.preprocessing import (
    charger_donnees_anonymisees,
    nettoyer_donnees,
    filtrer_contrats_eligibles,
    ajouter_variable_cible
)
from src.features import preparer_features
from src.eda import executer_eda_streamlit
from src.comparaison_models import comparer_modeles_streamlit
from src.training_xgboost import entrainer_xgboost
from src.Courbe_ROC import afficher_courbe_roc
from src.predict import predire_clients_a_risque

# ===== Configuration de la page =====
st.set_page_config(page_title="Dashboard LLD", layout="wide")

# ===== Styles personnalisés =====
st.markdown("""
    <style>
        .main { background-color: #f9f9fc; }
        header, footer { visibility: hidden; }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            color: gray;
            text-align: center;
            font-size: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# ===== Bandeau d'accueil =====
st.markdown(
    """
    <div style='background-color:#795d8d;padding:4px;border-radius:5px;margin-bottom:12px'>
        <h2 style='color:white;text-align:center;'>Tableau de bord de prédiction location longue durée </h2>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== Logo + Présentation =====
logo = Image.open("assets/voiture.jpg")
col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=85)
with col2:
    st.markdown(
        """
        <div style='font-size:20px; padding-top:10px;'>
            <b>Cette application</b> 
             vise à prédire les contrats professionnels à risque de non-renouvellement afin d’optimiser les actions de fidélisation.
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== Zone de chargement du fichier Excel (stylée) =====
st.markdown("---")
st.markdown("""
    <div style="background-color:#f0f2f6; padding:25px; border-radius:10px; margin-top:15px; margin-bottom:25px;">
        <h4 style='margin-bottom:15px;'>📂 Charger les données anonymisées</h4>
        <p style='color:#444;'>Veuillez importer le fichier Excel contenant les contrats anonymisés (format .xlsx, max 200MB).</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["xlsx"],
    help="Déposez ici le fichier contenant les données anonymisées au format Excel"
)

# ===== Traitement si le fichier est chargé =====
if uploaded_file:
    try:
        with st.spinner("Chargement et traitement des données, veuillez patienter..."):
            df = charger_donnees_anonymisees(uploaded_file)
            st.success("✅ Données chargées")

            df_cleaned = nettoyer_donnees(df)
            df_filtered = filtrer_contrats_eligibles(df_cleaned)
            df_final = ajouter_variable_cible(df_filtered)
            df_model = preparer_features(df_final)

        # ===== Navigation par onglets (sans entraînement) =====
        onglets = st.tabs([
            "🔍 Analyse exploratoire (EDA)",
            "📊 Comparaison modèles",
            "📈 Courbe ROC",
            "🚨 Clients à risque",
            "🏆 Top 50 clients à risque"
        ])

        # 🔍 Analyse exploratoire
        with onglets[0]:
            st.subheader("🔍 Analyse exploratoire des données")
            st.markdown("### 📊 Résumé global des données")

            # === KPI calculs ===
            total_contrats = len(df_final)
            contrats_actifs = df_final["flag_actif"].sum()
            non_renouveles = df_final["Non_renouvellement"].sum()
            taux_non_renouvellement = round((non_renouveles / total_contrats) * 100, 2)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📄 Total contrats", f"{total_contrats:,}".replace(",", " "))
            col2.metric("🟢 Contrats actifs", f"{contrats_actifs:,}".replace(",", " "))
            col3.metric("❌ Non renouvelés", f"{non_renouveles:,}".replace(",", " "))
            col4.metric("📉 % Non renouvelés", f"{taux_non_renouvellement} %")

            st.markdown("---")
            executer_eda_streamlit(df_model)

        # 📊 Comparaison modèles
        with onglets[1]:
            st.subheader("📊 Comparaison des modèles de prédiction")
            comparer_modeles_streamlit(df_model)

        # 📈 Courbe ROC
        with onglets[2]:
            st.subheader("📈 Courbe ROC - XGBoost")
            afficher_courbe_roc(df_model, model_path="models/xgboost_model.joblib")

        # 🚨 Clients à risque
        with onglets[3]:
            st.subheader("🚨 Prédiction des clients à risque de non-renouvellement")
            df_risque, df_top_50 = predire_clients_a_risque(df_model)
            st.success(f"{len(df_risque)} clients à risque détectés")

            recherche = st.text_input("🔎 Rechercher un contrat")
            if recherche:
                resultats = df_risque[df_risque["No du Contrat"].astype(str).str.contains(recherche)]
            else:
                resultats = df_risque.head(1000)

            st.write("📋 Liste complète (limitée à 1000 lignes)")
            st.dataframe(resultats, use_container_width=True)

            buffer_risque = io.BytesIO()
            with pd.ExcelWriter(buffer_risque, engine="openpyxl") as writer:
                df_risque.to_excel(writer, sheet_name="clients_a_risque", index=False)

            st.download_button(
                label="📥 Télécharger tous les clients à risque (Excel)",
                data=buffer_risque.getvalue(),
                file_name="clients_a_risque.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 🏆 Top 50 clients
        with onglets[4]:
            st.subheader("🏆 Top 50 clients à plus haut risque")
            st.write("📈 Ces clients présentent le risque le plus élevé de non-renouvellement.")
            st.dataframe(df_top_50, use_container_width=True)

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_top_50.to_excel(writer, sheet_name="top_50_clients", index=False)

            st.download_button(
                label="📥 Télécharger le Top 50 (Excel)",
                data=buffer.getvalue(),
                file_name="top_50_clients.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")

else:
    st.info("💡 Veuillez charger un fichier Excel anonymisé pour commencer.")

