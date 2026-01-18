import pandas as pd

# Anonymisation des données
def charger_donnees_anonymisees(fichier: str) -> pd.DataFrame:
    """
    Charge les données anonymisées depuis un fichier Excel.
    """
    return pd.read_excel(fichier)

# Nettoyage des données
# Applique les premières étapes de nettoyage : suppression des doublons, lignes vides, nettoyage des noms de colonnes,
def nettoyer_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique les premières étapes de nettoyage :
    - Suppression des doublons
    - Suppression des lignes entièrement vides
    - Nettoyage des noms de colonnes
    - Standardisation des chaînes de caractères
    """
    df = df.copy()

    # Supprimer les doublons exacts
    df.drop_duplicates(inplace=True)

    # Supprimer les lignes entièrement vides
    df.dropna(how="all", inplace=True)

    # Nettoyage des noms de colonnes (espaces en trop)
    df.columns = [col.strip() for col in df.columns]

    # Standardiser certaines colonnes texte
    for col in ["Type Commande", "Nouveau Client"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df

# Filtrage des contrats éligibles pour la prédiction du non-renouvellement
# On garde uniquement les contrats pertinents pour la prédiction du non-renouvellement
## - Garde uniquement les clients existants (Nouveau Client == NON)
## - Exclut les nouvelles commandes
# Cette étape est cruciale pour s'assurer que seuls les contrats pertinents sont analysés   

def filtrer_contrats_eligibles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les contrats pertinents pour la prédiction du non-renouvellement :
    - Garde uniquement les clients existants (Nouveau Client == NON)
    - Exclut les nouvelles commandes
    """

    df_filtre = df.copy()

    # Standardiser pour éviter les erreurs de casse/espaces
    if "Nouveau Client" in df_filtre.columns:
        df_filtre["Nouveau Client"] = df_filtre["Nouveau Client"].astype(str).str.strip().str.upper()

    if "Type Commande" in df_filtre.columns:
        df_filtre["Type Commande"] = df_filtre["Type Commande"].astype(str).str.strip().str.lower()

    # Appliquer le filtre métier
    df_filtre = df_filtre[
        (df_filtre["Nouveau Client"] == "NON") &
        (df_filtre["Type Commande"] != "nouvelle commande")
    ]

    return df_filtre


# Ajout de la variable cible pour la prédiction du non-renouvellement
# Crée une colonne 'Non_renouvellement' :
# - 0 si Type Commande = renouvellement
# - 1 sinon (extension de parc ou autre)

def ajouter_variable_cible(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une colonne 'Non_renouvellement' :
    - 0 si Type Commande = renouvellement
    - 1 sinon (extension de parc ou autre)
    """
    df_copy = df.copy()

    if "Type Commande" in df_copy.columns:
        type_commande = df_copy["Type Commande"].astype(str).str.strip().str.lower()
        df_copy["Non_renouvellement"] = type_commande.apply(
            lambda x: 0 if x == "renouvellement" else 1
        )
    else:
        raise ValueError("La colonne 'Type Commande' est introuvable.")

    return df_copy
