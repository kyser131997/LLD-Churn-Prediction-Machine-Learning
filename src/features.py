import pandas as pd

def preparer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les variables explicatives optimisées pour la modélisation :
    - Garde uniquement les variables pertinentes selon EDA
    - Calcule l'ancienneté du contrat (mois)
    - Calcule l'écart de restitution (jours)
    - Encode les prestations discriminantes
    - Conserve 'No du Contrat' uniquement pour traçabilité
    """
    df_features = df.copy()

    # 🔍 1. Nettoyage des valeurs manquantes
    if "Vendeur Réseau" in df_features.columns:
        df_features["Vendeur Réseau"] = df_features["Vendeur Réseau"].fillna("Inconnu")

    if "Montant mise à la route" in df_features.columns:
        mediane = df_features["Montant mise à la route"].median()
        df_features["Montant mise à la route"] = df_features["Montant mise à la route"].fillna(mediane)

    # 📅 2. Calcul de l'ancienneté du contrat (en mois)
    df_features["Date de Commande"] = pd.to_datetime(df_features["Date de Commande"], errors="coerce", dayfirst=True)
    df_features["Date de fin du contrat"] = pd.to_datetime(df_features["Date de fin du contrat"], errors="coerce", dayfirst=True)

    df_features["Anciennete_contrat"] = (
        (df_features["Date de fin du contrat"].dt.year - df_features["Date de Commande"].dt.year) * 12 +
        (df_features["Date de fin du contrat"].dt.month - df_features["Date de Commande"].dt.month)
    )
    df_features = df_features[df_features["Anciennete_contrat"].between(1, 120)]

    # 📦 3. Ecart restitution
    df_features["Date de restitution"] = pd.to_datetime(df_features["Date de restitution"], errors="coerce", dayfirst=True)
    df_features.loc[df_features["Date de restitution"].dt.year < 2000, "Date de restitution"] = pd.NaT
    df_features["Date de restitution"] = df_features["Date de restitution"].fillna(df_features["Date de fin du contrat"])
    df_features["Ecart_restitution_jours"] = (
        df_features["Date de restitution"] - df_features["Date de fin du contrat"]
    ).dt.days
    df_features["Ecart_restitution_jours"] = df_features["Ecart_restitution_jours"].fillna(0)

    # 🔢 4. Encodage des prestations discriminantes
    for col in ["Gest. carburant", "Assurance", "Divers"]:
        if col in df_features.columns:
            df_features[col + "_bin"] = df_features[col].str.upper().map({"OUI": 1, "NON": 0})

    # ✅ 5. Sélection des variables finales (avec traçabilité)
    colonnes_finales = [
        "No du Contrat",                 # pour traçabilité uniquement
        "Non_renouvellement",   
        "flag_actif",        # cible
        "Anciennete_contrat",
        "Ecart_restitution_jours",
        "Montant loyer mensuel",
        "Km souscrit",
        "Nombre de prestations",
        "Gest. carburant_bin",
        "Assurance_bin",
        "Divers_bin"
    ]

    df_model = df_features[[col for col in colonnes_finales if col in df_features.columns]].copy()

    return df_model
