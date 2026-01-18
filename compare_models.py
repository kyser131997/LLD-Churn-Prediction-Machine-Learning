import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# 📁 Charger les données préparées
df = pd.read_excel("data/processed/donnees_finales_model.xlsx")

# 🔀 Séparer X et y
X = df.drop(columns=["Non_renouvellement", "No du Contrat"])
y = df["Non_renouvellement"]

# ✂️ Train/test split
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

# 📊 Stocker les résultats
resultats = []

for nom, modele in modeles.items():
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f"\n📈 {nom}")
    print(classification_report(y_test, y_pred))

    resultats.append({
        "Modèle": nom,
        "F1-score": round(f1, 4),
        "Recall (classe 1)": round(recall, 4),
        "Précision": round(precision, 4)
    })

# 💾 Sauvegarder le résumé
df_resultats = pd.DataFrame(resultats)
os.makedirs("outputs/rapports", exist_ok=True)
df_resultats.to_csv("outputs/rapports/model_comparison.csv", index=False)

print("\n✅ Comparaison enregistrée dans outputs/rapports/model_comparison.csv")
