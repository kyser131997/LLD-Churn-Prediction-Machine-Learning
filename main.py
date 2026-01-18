import pandas as pd
from src.preprocessing import (
    nettoyer_donnees,
    filtrer_contrats_eligibles,
    ajouter_variable_cible
)
from src.features import preparer_features
from src.eda import executer_eda
from src.tests_statistiques import executer_tests_statistiques
import os

# 📥 1. Charger les données anonymisées
df = pd.read_excel("data/processed/donnees_anonymisees.xlsx")

# 🧼 2. Nettoyer les données
df = nettoyer_donnees(df)

# 🔍 3. Filtrer les contrats éligibles
df = filtrer_contrats_eligibles(df)

# 🎯 4. Ajouter la variable cible
df = ajouter_variable_cible(df)

# 🧠 5. Préparer les features (avec ID pour traçabilité)
df_model = preparer_features(df)

# 💾 6. Sauvegarder le jeu final pour modélisation
os.makedirs("data/processed", exist_ok=True)
df_model.to_excel("data/processed/donnees_finales_model.xlsx", index=False)

# 📊 7. Lancer EDA
executer_eda(df_model)

# 🧪 8. Lancer les tests statistiques
executer_tests_statistiques(df_model)

print("\n✅ Pipeline terminé : Données prêtes, EDA et tests statistiques générés.")
