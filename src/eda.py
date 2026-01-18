import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

FIGURES_PATH = "outputs/figures"
RAPPORTS_PATH = "outputs/rapports"
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(RAPPORTS_PATH, exist_ok=True)

import streamlit as st

def executer_eda_streamlit(df):
    st.markdown("## 📊 Analyse exploratoire des données")

    # Dimensions
    st.subheader("📏 Statistiques générales")
    st.write(f"*Dimensions :* {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # Types de données (affichage horizontal)
    st.write("*Types de données :*")
    types_df = pd.DataFrame(df.dtypes, columns=["Type"]).T
    st.dataframe(types_df)

    # Valeurs manquantes
    st.subheader("❓ Valeurs manquantes")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Colonne", "Valeurs manquantes"]
    missing["Valeurs manquantes"] = missing["Valeurs manquantes"].astype(int)
    st.dataframe(missing)

    st.markdown("## 🧠 Analyse graphique")

    # Liste des noms de fichiers (ordre voulu)
    noms_figures = [
        "Non_renouvellement_distribution.png",
        "Anciennete_contrat_vs_Non_renouvellement.png",
        "Km souscrit_vs_Non_renouvellement.png",
        "Montant loyer mensuel_vs_Non_renouvellement.png",
        "Ecart_restitution_jours_vs_Non_renouvellement.png",
        "Assurance_bin_vs_Non_renouvellement.png",
        "Gest. carburant_bin_vs_Non_renouvellement.png",
        "Divers_bin_vs_Non_renouvellement.png",
        "correlation_matrix.png"
    ]

    FIGURES_PATH = "outputs/figures"

    # Affichage par blocs de 3
    for i in range(0, len(noms_figures), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(noms_figures):
                fig_path = os.path.join(FIGURES_PATH, noms_figures[i + j])
                if os.path.exists(fig_path):
                    with cols[j]:
                        st.image(fig_path, use_container_width=True, caption=noms_figures[i + j].replace("_", " ").replace(".png", ""))