"""
Módulo para Balanceamento de Dados
Responsável por detectar desequilíbrio e aplicar técnicas de balanceamento (SMOTE, etc).
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


def check_class_imbalance(y: pd.Series, threshold: float = 0.4) -> Dict[str, Any]:
    """
    Verifica se há desequilíbrio significativo de classes.

    Args:
        y (pd.Series): Série com a variável alvo
        threshold (float): Limite para considerar desequilíbrio (proporção mínima)

    Returns:
        dict: Informações sobre desequilíbrio
    """
    class_dist = y.value_counts(normalize=True)
    min_proportion = class_dist.min()
    has_imbalance = min_proportion < threshold

    return {
        'distribution': class_dist.to_dict(),
        'min_class_proportion': min_proportion,
        'has_imbalance': has_imbalance,
        'imbalance_ratio': class_dist.max() / class_dist.min()
    }


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Aplica SMOTE para balanceamento sobre a classe minoritária.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Variável alvo
        random_state (int): Seed para reprodutibilidade

    Returns:
        tuple: (X_balanced, y_balanced)
    """
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Converter de volta para DataFrame se necessário
    if isinstance(X, pd.DataFrame):
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)

    if isinstance(y, pd.Series):
        y_balanced = pd.Series(y_balanced, name=y.name)

    return X_balanced, y_balanced


def compute_class_weights(y: pd.Series) -> Dict[str, float]:
    """
    Calcula pesos para cada classe para balanceamento em treino.
    Útil em modelos que suportam 'class_weight'.

    Args:
        y (pd.Series): Variável alvo

    Returns:
        dict: Pesos de classe
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)

    return {str(cls): float(weight) for cls, weight in zip(classes, weights)}


def oversample_minority_class(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Oversampling simples da classe minoritária (duplicação aleatória).

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Variável alvo

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    from sklearn.utils import resample

    X_copy = X.copy()
    y_copy = y.copy()

    # Identificar classes
    classes = np.unique(y_copy)
    minority_class = classes[0] if (y_copy == classes[0]).sum() < (
        y_copy == classes[1]).sum() else classes[1]

    # Separar minoritárias e majoritárias
    minority_idx = (y_copy == minority_class).values
    majority_idx = ~minority_idx

    X_minority = X_copy[minority_idx]
    X_majority = X_copy[majority_idx]

    y_minority = y_copy[minority_idx]
    y_majority = y_copy[majority_idx]

    # Resample minoritárias para igualar
    X_minority_resampled = resample(
        X_minority,
        n_samples=len(X_majority),
        replace=True
    )
    y_minority_resampled = pd.Series(
        [minority_class] * len(X_majority),
        name=y.name
    )

    # Combinar
    X_resampled = pd.concat([X_majority.reset_index(drop=True),
                             X_minority_resampled.reset_index(drop=True)],
                            ignore_index=True)
    y_resampled = pd.concat([y_majority.reset_index(drop=True),
                             y_minority_resampled.reset_index(drop=True)],
                            ignore_index=True)

    return X_resampled, y_resampled
