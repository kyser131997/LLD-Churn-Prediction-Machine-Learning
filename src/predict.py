import pandas as pd
import joblib
import numpy as np

def predire_clients_a_risque(df: pd.DataFrame, model_path: str = "models/xgboost_model.joblib"):
    """
    Prédiction des contrats actifs à risque (non-renouvelés) via le modèle XGBoost.
    Retourne :
        - df_risque : Tous les contrats prédits comme non-renouvelés
        - df_top_50 : Les 50 clients à plus haut risque (score)
    """

    # 🛑 Vérifier la colonne "flag_actif"
    if "flag_actif" not in df.columns:
        raise ValueError("La colonne 'flag_actif' est manquante dans le DataFrame.")

    # 🔍 1. Filtrer les contrats actifs
    df_actifs = df[df["flag_actif"] == 1].copy()

    # 🧠 2. Charger le modèle
    model = joblib.load(model_path)

    # 📋 3. Définir les features attendues par le modèle
    expected_features = model.get_booster().feature_names

    # 🧼 4. Préparer X (et ajouter les colonnes manquantes si besoin)
    X = df_actifs.drop(columns=["No du Contrat", "Non_renouvellement", "flag_actif"], errors="ignore").copy()

    # Ajouter colonnes manquantes
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0  # ou np.nan selon ton choix

    # Réordonner les colonnes dans l'ordre attendu
    X = X[expected_features]

    # 🔮 5. Prédictions
    df_actifs["Prediction"] = model.predict(X)
    df_actifs["score_risque"] = model.predict_proba(X)[:, 1]

    # 🎯 6. Sélection des clients à risque
    df_risque = df_actifs[df_actifs["Prediction"] == 1].copy()
    df_top_50 = df_risque.sort_values("score_risque", ascending=False).head(50)

    return df_risque, df_top_50