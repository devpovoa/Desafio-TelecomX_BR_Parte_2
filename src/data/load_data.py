from pathlib import Path

import pandas as pd


def load_data():
    """
    Carrega o dataset tratado da TelecomX.
    """

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "dados_tratados.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado em: {data_path}"
        )

    df = pd.read_csv(data_path)

    return df
