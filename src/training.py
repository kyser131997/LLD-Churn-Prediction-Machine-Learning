import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def entrainer_random_forest(df_model: pd.DataFrame):
    """
    Entraîne un modèle Random Forest et le sauvegarde.
    """

    # 🔀 Séparer X et y (sans flag_actif)
    X = df_model.drop(columns=["No du Contrat", "Non_renouvellement", "flag_actif"], errors="ignore")
    y = df_model["Non_renouvellement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # 🧾 Rapport
    rapport = classification_report(y_test, y_pred)
    print("\n📈 Rapport Random Forest :")
    print(rapport)

    # 📊 Matrice de confusion
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion - Random Forest")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig("outputs/figures/confusion_matrix_random_forest.png")
    plt.close()

    # 💾 Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/random_forest_model.pkl")

    print("✅ Modèle Random Forest sauvegardé dans models/random_forest_model.pkl")