import pandas as pd
import hashlib

def hash_column(column: pd.Series) -> pd.Series:
    """
    Anonymise une colonne en utilisant le hachage SHA256.
    """
    return column.astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

def anonymiser_dataframe(df: pd.DataFrame, colonnes_sensibles: list) -> pd.DataFrame:
    """
    Applique l’anonymisation aux colonnes sensibles d’un DataFrame.

    Args:
        df : DataFrame contenant les données originales
        colonnes_sensibles : liste des colonnes à anonymiser

    Returns:
        Un DataFrame avec les colonnes spécifiées anonymisées
    """
    df_copy = df.copy()
    for col in colonnes_sensibles:
        if col in df_copy.columns:
            df_copy[col] = hash_column(df_copy[col])
    return df_copy
