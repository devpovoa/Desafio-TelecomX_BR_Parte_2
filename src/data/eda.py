"""
Módulo para Análise Exploratória de Dados (EDA)
Responsável por carregar, explorar e visualizar os dados brutos tratados.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV de dados tratados.

    Args:
        filepath (str): Caminho para o arquivo CSV

    Returns:
        pd.DataFrame: DataFrame com os dados carregados
    """
    df = pd.read_csv(filepath)
    return df


def basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Fornece informações básicas sobre o dataset.

    Args:
        df (pd.DataFrame): DataFrame para análise

    Returns:
        dict: Dicionário com informações do dataset
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    return info


def analyze_target_variable(df: pd.DataFrame, target_col: str = 'churn') -> Dict[str, Any]:
    """
    Analisa a variável alvo (churn) - distribuição e desequilíbrio.

    Args:
        df (pd.DataFrame): DataFrame com dados
        target_col (str): Nome da coluna de churn

    Returns:
        dict: Análise de distribuição da classe alvo
    """
    if target_col not in df.columns:
        raise ValueError(f"Coluna '{target_col}' não encontrada no DataFrame")

    # Remove valores nulos
    target_clean = df[target_col].dropna()

    analysis = {
        'distribution': target_clean.value_counts().to_dict(),
        'proportions': target_clean.value_counts(normalize=True).to_dict(),
        'imbalance_ratio': target_clean.value_counts(normalize=True).min()
    }

    return analysis


def numeric_features_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna um resumo das variáveis numéricas.

    Args:
        df (pd.DataFrame): DataFrame para análise

    Returns:
        pd.DataFrame: Resumo de variáveis numéricas
    """
    return df.select_dtypes(include=[np.number]).describe()


def categorical_features_summary(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Retorna um resumo das variáveis categóricas.

    Args:
        df (pd.DataFrame): DataFrame para análise

    Returns:
        dict: Resumo de variáveis categóricas
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    summary = {}

    for col in categorical_cols:
        summary[col] = df[col].value_counts().to_dict()

    return summary


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verifica a qualidade dos dados (valores nulos, duplicatas, tipos).

    Args:
        df (pd.DataFrame): DataFrame para análise

    Returns:
        dict: Relatório de qualidade dos dados
    """
    quality_report = {
        'total_rows': len(df),
        'null_values': df.isnull().sum().sum(),
        'null_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'duplicates': df.duplicated().sum(),
        'columns_with_nulls': df.columns[df.isnull().any()].tolist(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
    }

    return quality_report
