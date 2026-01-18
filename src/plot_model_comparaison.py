import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 📥 Charger les résultats
df = pd.read_csv("outputs/rapports/model_comparison.csv")

# 🎨 Configuration du style
sns.set(style="whitegrid")

# 📊 Tracer le barplot comparatif
plt.figure(figsize=(10, 5))
df.set_index("Modèle")[["F1-score", "Recall (classe 1)", "Précision"]].plot(
    kind="bar", figsize=(10, 5), colormap="viridis", ylim=(0, 1)
)
plt.title("Comparaison des modèles de prédiction")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

# 💾 Sauvegarder
os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/model_comparaison.png")
plt.close()

print("✅ Graphique de comparaison enregistré dans outputs/figures/model_comparison.png")
