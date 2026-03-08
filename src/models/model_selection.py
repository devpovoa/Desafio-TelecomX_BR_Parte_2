"""
Módulo para Seleção e Treinamento de Modelos
Responsável por construir, treinar e avaliar modelos preditivos de churn.
"""

from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class ModelEvaluator:
    """
    Classe para avaliar modelos de classificação com múltiplas métricas.
    """

    @staticmethod
    def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Avalia um modelo treinado com múltiplas métricas.

        Args:
            model: Modelo treinado
            X_test (pd.DataFrame): Features de teste
            y_test (pd.Series): Alvo de teste

        Returns:
            dict: Métricas de avaliação
        """
        y_pred = model.predict(X_test)

        # Para modelos que retornam probabilidades
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        return metrics

    @staticmethod
    def compare_models(models_dict: Dict[str, Any], X_test: pd.DataFrame,
                       y_test: pd.Series) -> pd.DataFrame:
        """
        Compara múltiplos modelos lado a lado.

        Args:
            models_dict (dict): Dicionário {nome_modelo: modelo_treinado}
            X_test (pd.DataFrame): Features de teste
            y_test (pd.Series): Alvo de teste

        Returns:
            pd.DataFrame: Comparação de métricas
        """
        results = []

        for name, model in models_dict.items():
            metrics = ModelEvaluator.evaluate_model(model, X_test, y_test)
            result = {
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc']
            }
            results.append(result)

        return pd.DataFrame(results)


class ModelFactory:
    """
    Factory para criar e treinar modelos de classificação.
    """

    @staticmethod
    def build_logistic_regression(random_state: int = 42, **kwargs) -> LogisticRegression:
        """Constrói um modelo de Regressão Logística."""
        return LogisticRegression(random_state=random_state, max_iter=1000, **kwargs)

    @staticmethod
    def build_random_forest(random_state: int = 42, **kwargs) -> RandomForestClassifier:
        """Constrói um modelo de Random Forest."""
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )

    @staticmethod
    def build_svm(kernel: str = 'rbf', **kwargs) -> SVC:
        """Constrói um modelo de SVM."""
        return SVC(kernel=kernel, probability=True, **kwargs)

    @staticmethod
    def build_knn(n_neighbors: int = 5, **kwargs) -> KNeighborsClassifier:
        """Constrói um modelo de KNN."""
        return KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)


class ModelTrainer:
    """
    Classe para treinar modelos com cross-validation e ajuste de hiperparâmetros.
    """

    @staticmethod
    def train_model(model, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Treina um modelo.

        Args:
            model: Modelo (sklearn)
            X_train (pd.DataFrame): Features de treino
            y_train (pd.Series): Alvo de treino

        Returns:
            Modelo treinado
        """
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                             cv: int = 5) -> Dict[str, Any]:
        """
        Realiza cross-validation de um modelo.

        Args:
            model: Modelo (sklearn)
            X_train (pd.DataFrame): Features de treino
            y_train (pd.Series): Alvo de treino
            cv (int): Número de folds

        Returns:
            dict: Scores de cross-validation
        """
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='f1_weighted')

        return {
            'cv_scores': scores,
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std()
        }


def save_model(model, filepath: str) -> None:
    """
    Salva um modelo treinado em arquivo.

    Args:
        model: Modelo treinado
        filepath (str): Caminho do arquivo
    """
    joblib.dump(model, filepath)
    print(f"Modelo salvo em: {filepath}")


def load_model(filepath: str):
    """
    Carrega um modelo salvo.

    Args:
        filepath (str): Caminho do arquivo

    Returns:
        Modelo carregado
    """
    return joblib.load(filepath)
