import pandas as pd


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as transformações necessárias para preparar dados para ML.
    
    Passos:
    1. Remove coluna customerid
    2. Remove linhas com churn nulo
    3. Converte churn de Yes/No para 1/0
    4. Trata valores nulos em colunas numéricas
    5. Aplica One-Hot Encoding em variáveis categóricas
    """
    
    df = df.copy()

    # 1. Remover ID do cliente (não preditivo)
    df = df.drop(columns=["customerid"], errors="ignore")

    # 2. Remover linhas sem target (churn nulo)
    df = df.dropna(subset=["churn"])

    # 3. Converter target para numérico (Yes=1, No=0)
    if df["churn"].dtype == "object" or df["churn"].dtype == "string":
        # Usar replace em vez de map para maior robustez
        df["churn"] = df["churn"].replace({"Yes": 1, "No": 0})
        # Garantir que converteu corretamente
        df["churn"] = pd.to_numeric(df["churn"], errors='coerce')
        # Se ainda houver NaNs, remover
        df = df.dropna(subset=["churn"])
        df["churn"] = df["churn"].astype(int)

    # 4. Tratar missed values em colunas numéricas
    if "account_charges_total" in df.columns:
        df["account_charges_total"] = df["account_charges_total"].fillna(
            df["account_charges_total"].median()
        )

    # 5. Identificar colunas categóricas (exceto churn, já convertida)
    categorical_columns = [
        col for col in df.select_dtypes(include=["object"]).columns 
        if col != "churn"
    ]

    # Aplicar One-Hot Encoding
    if categorical_columns:
        df_encoded = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True,
            dtype=bool
        )
    else:
        df_encoded = df

    return df_encoded
