import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
import os

RAPPORTS_PATH = "outputs/rapports/"
os.makedirs(RAPPORTS_PATH, exist_ok=True)

def test_chi2(df: pd.DataFrame, var_cat: str, cible: str = "Non_renouvellement"):
    """Test du chi² pour une variable binaire vs la cible"""
    table = pd.crosstab(df[var_cat], df[cible])
    chi2, p, dof, expected = chi2_contingency(table)
    return {"variable": var_cat, "chi2": chi2, "p_value": p}

def test_ttest(df: pd.DataFrame, var_cont: str, cible: str = "Non_renouvellement"):
    """Test de Student pour une variable continue vs la cible"""
    group0 = df[df[cible] == 0][var_cont].dropna()
    group1 = df[df[cible] == 1][var_cont].dropna()
    stat, p = ttest_ind(group0, group1, equal_var=False)
    return {"variable": var_cont, "statistic": stat, "p_value": p}

def executer_tests_statistiques(df: pd.DataFrame):
    rapport_path = os.path.join(RAPPORTS_PATH, "tests_statistiques.txt")
    with open(rapport_path, "w", encoding="utf-8") as f:

        f.write("📊 TESTS STATISTIQUES\n\n")

        # 🔢 Chi² pour variables binaires
        variables_chi2 = ["Assurance_bin", "Gest. carburant_bin", "Divers_bin"]
        f.write("🔍 Test du Chi² (variables binaires vs Non_renouvellement)\n")
        for var in variables_chi2:
            if var in df.columns:
                result = test_chi2(df, var)
                f.write(f"{result['variable']} : p-value = {result['p_value']:.4f}\n")
        f.write("\n")

        # 📉 T-test pour variables continues
        variables_continues = [
            "Montant loyer mensuel",
            "Km souscrit",
            "Anciennete_contrat",
            "Ecart_restitution_jours"
        ]
        f.write("📉 Test de Student (variables continues vs Non_renouvellement)\n")
        for var in variables_continues:
            if var in df.columns:
                result = test_ttest(df, var)
                f.write(f"{result['variable']} : p-value = {result['p_value']:.4f}\n")

    print(f"✅ Tests statistiques enregistrés dans {rapport_path}")
