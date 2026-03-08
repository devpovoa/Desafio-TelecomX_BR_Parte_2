"""
Módulo para Visualização de Correlação
Responsável por criar gráficos de correlação e análise de variáveis.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (14, 10),
                            cmap: str = 'coolwarm', title: str = 'Matriz de Correlação') -> None:
    """
    Cria um heatmap com a matriz de correlação de variáveis numéricas.

    Args:
        df (pd.DataFrame): DataFrame com dados (apenas colunas numéricas serão consideradas)
        figsize (tuple): Tamanho da figura
        cmap (str): Mapa de cor
        title (str): Título do gráfico
    """
    # Selecionar apenas variáveis numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    # Calcular correlação
    correlation = numeric_df.corr()

    # Criar figura
    plt.figure(figsize=figsize)
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap=cmap,
                square=True, cbar_kws={'label': 'Correlação'},
                vmin=-1, vmax=1, center=0)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_target_correlation(df: pd.DataFrame, target_col: str = 'churn',
                            numeric_cols: Optional[List[str]] = None,
                            figsize: tuple = (12, 6)) -> pd.Series:
    """
    Plota a correlação de variáveis numéricas com a variável alvo.

    Args:
        df (pd.DataFrame): DataFrame com dados
        target_col (str): Nome da coluna alvo
        numeric_cols (list): Colunas numéricas a correlacionar (se None, seleciona todas)
        figsize (tuple): Tamanho da figura

    Returns:
        pd.Series: Correlação com a variável alvo
    """
    # Converter target para numérico se necessário
    if df[target_col].dtype == 'object':
        target_numeric = (df[target_col] == 'Yes').astype(int)
    else:
        target_numeric = df[target_col]

    # Selecionar colunas numéricas
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remover a coluna alvo se estiver incluída
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

    # Calcular correlação
    correlations = pd.DataFrame({
        'Feature': numeric_cols,
        'Correlation': [df[col].corr(target_numeric) for col in numeric_cols]
    })

    correlations = correlations.sort_values(
        'Correlation', key=abs, ascending=False)

    # Criar gráfico
    plt.figure(figsize=figsize)
    colors = ['green' if x > 0 else 'red' for x in correlations['Correlation']]
    plt.barh(correlations['Feature'],
             correlations['Correlation'], color=colors, alpha=0.7)
    plt.xlabel('Correlação com Churn', fontsize=12)
    plt.title('Correlação de Variáveis Numéricas com Churn',
              fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    plt.show()

    return correlations.set_index('Feature')['Correlation']


def plot_feature_importance(model, feature_names: List[str], top_n: int = 15,
                            figsize: tuple = (10, 8)) -> None:
    """
    Plota a importância das features de um modelo (Random Forest, XGBoost, etc).

    Args:
        model: Modelo que possui atributo feature_importances_
        feature_names (list): Nomes das features
        top_n (int): Número de top features a exibir
        figsize (tuple): Tamanho da figura
    """
    # Obter importância das features
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    # Criar gráfico
    plt.figure(figsize=figsize)
    plt.barh(range(len(indices)),
             importances[indices], alpha=0.7, color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importância', fontsize=12)
    plt.title(f'Top {top_n} Features - Importância do Modelo',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_logistic_coefficients(model, feature_names: List[str], top_n: int = 15,
                               figsize: tuple = (10, 8)) -> None:
    """
    Plota os coeficientes de um modelo de Regressão Logística.

    Args:
        model: Modelo de Regressão Logística treinado
        feature_names (list): Nomes das features
        top_n (int): Número de top coeficientes a exibir
        figsize (tuple): Tamanho da figura
    """
    # Obter coeficientes
    coefs = model.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    })
    coef_df['Abs_Coeff'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coeff', ascending=False).head(top_n)

    # Criar gráfico
    plt.figure(figsize=figsize)
    colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'],
             color=colors, alpha=0.7)
    plt.xlabel('Coeficiente', fontsize=12)
    plt.title(f'Top {top_n} Coeficientes - Regressão Logística',
              fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(models_dict: dict, X_test: pd.DataFrame,
                            y_test: pd.Series, figsize: tuple = (15, 5)) -> None:
    """
    Plota matrizes de confusão para múltiplos modelos.

    Args:
        models_dict (dict): Dicionário {nome: modelo_treinado}
        X_test (pd.DataFrame): Features de teste
        y_test (pd.Series): Alvo de teste
        figsize (tuple): Tamanho da figura
    """
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, len(models_dict), figsize=figsize)

    if len(models_dict) == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    cbar=False, annot_kws={'size': 14})
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Real', fontsize=10)
        axes[idx].set_xlabel('Predito', fontsize=10)

    plt.tight_layout()
    plt.show()
