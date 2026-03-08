import pandas as pd


def encode_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # remover ID
    df = df.drop(columns=["customerid"], errors="ignore")

    # remover linhas sem target
    df = df.dropna(subset=["churn"])

    # converter target se necessário
    if df["churn"].dtype == "object":
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    # tratar missing
    df["account_charges_total"] = df["account_charges_total"].fillna(
        df["account_charges_total"].median()
    )

    # colunas categóricas
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # one-hot encoding
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    return df_encoded
