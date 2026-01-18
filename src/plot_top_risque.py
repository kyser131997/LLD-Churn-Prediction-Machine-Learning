import matplotlib.pyplot as plt
import seaborn as sns

def afficher_top_clients(df_top_50):
    """
    Affiche le top 10 des clients les plus à risque (score de non-renouvellement élevé)
    """
    plt.figure(figsize=(10, 6))
    df_visu = df_top_50.head(10).copy()
    df_visu["Contrat"] = df_visu["No du Contrat"].astype(str)

    sns.barplot(
        data=df_visu,
        x="score_risque",
        y="Contrat",
        palette="Reds_r"
    )

    for i, score in enumerate(df_visu["score_risque"]):
        plt.text(score + 0.001, i, f"{score:.3f}", va="center")

    plt.title("Top 10 des clients les plus à risque")
    plt.xlabel("Score de risque (proba de non-renouvellement)")
    plt.ylabel("Contrat")
    plt.tight_layout()
    plt.show()