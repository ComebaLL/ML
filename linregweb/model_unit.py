__author__ = "Kuvykin Nikita"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any, Dict

# Получение данных из файла .csv

class ITrainingDataProvider(ABC):
    """Интерфейс для поставщика обучающих данных."""
    @abstractmethod
    def get_data(self) -> pd.DataFrame:
        pass

class CsvTrainingDataProvider(ITrainingDataProvider):
    """
    Загружает данные из CSV файла.
    """
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._cache = None

    def get_data(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache
            
        try:
            self._cache = pd.read_csv(self._file_path)
            return self._cache
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл данных не найден: {self._file_path}")
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения данных: {e}")


# Класс для создания формулы линейной регрессии

class FormulaFormatter:
    """
    Отвечает только за создание строкового представления формулы (LaTeX).
    """
    @staticmethod
    def format_linear_equation(coefs: np.ndarray, intercept: float) -> str:
        """Генерирует LaTeX строку для уравнения линейной регрессии."""
        parts = []
        
        # Свободный член
        if abs(intercept) > 0.001:
            parts.append(f"{intercept:.4f}")

        # Коэффициенты
        for i, coef in enumerate(coefs):
            if abs(coef) > 0.001:
                sign = " + " if coef >= 0 else " - "
                parts.append(f"{sign}{abs(coef):.4f} x_{{{i+1}}}")
        
        if not parts:
            return "y = 0"
            
        equation = "".join(parts)
        # Убираем начальный плюс, если он есть
        if equation.startswith(" + "):
            equation = equation[3:]
            
        return f"y = {equation}"


# Класс для получения метрик и сатитстики модели

class ModelAnalyzer:
    """
    Отвечает за извлечение метрик и статистики из модели.
    Не знает про графики и файлы, только про модель и DataFrame.
    """
    def __init__(self, model: Any, data_provider: ITrainingDataProvider):
        self._model = model
        self._data_provider = data_provider
        self._feature_cols = ['x1', 'x2', 'x3', 'x4'] 
        self._target_col = 'y'

    def get_r2_score(self) -> str:
        try:
            data = self._data_provider.get_data()
            X = data[self._feature_cols]
            y = data[self._target_col]
            
            if not hasattr(self._model, 'score'):
                return "Модель не поддерживает метод score"
                
            r2 = self._model.score(X, y)
            return f"{r2:.4f}"
        except Exception as e:
            return f"Ошибка расчета R2: {e}"

    def get_coefficients_array(self) -> Optional[np.ndarray]:
        if not hasattr(self._model, 'coef_') or not hasattr(self._model, 'intercept_'):
            return None
        return np.concatenate([[self._model.intercept_], self._model.coef_])

    def get_top_features_indices(self, n: int = 2) -> List[int]:
        """Возвращает индексы n самых важных признаков."""
        if not hasattr(self._model, 'coef_'):
            return [0, 1] # Fallback
        
        # Сортировка по модулю коэффициента (от большего к меньшему)
        return np.argsort(np.abs(self._model.coef_))[-n:][::-1].tolist()
    
    def get_feature_names(self) -> List[str]:
        return self._feature_cols


# Класс для создания графиков

class RegressionVisualizer:
    """
    Отвечает только за построение графиков.
    Использует ModelAnalyzer для получения нужных данных.
    """
    def __init__(self, analyzer: ModelAnalyzer):
        self._analyzer = analyzer

    def create_scatter_plots(self, user_input: np.ndarray, prediction: float) -> plt.Figure:
        data = self._analyzer._data_provider.get_data() # Доступ к данным через провайдер
        top_indices = self._analyzer.get_top_features_indices(n=2)
        feature_names = self._analyzer.get_feature_names()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Диаграммы рассеяния: Важнейшие признаки vs Целевая переменная')
        
        for idx, (ax, feature_idx) in enumerate(zip(axes, top_indices)):
            feat_name = feature_names[feature_idx]
            
            # График обучающих данных
            sns.scatterplot(data=data, x=feat_name, y='y', ax=ax, alpha=0.5, label='Training Data')
            sns.regplot(data=data, x=feat_name, y='y', ax=ax, scatter=False, color='blue')
            
            # Точка пользователя
            ax.scatter(user_input[feature_idx], prediction, color='red', s=100, 
                       label='Your Data', edgecolors='black', zorder=5)
            
            ax.set_xlabel(feat_name)
            ax.set_ylabel('Target (y)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        return fig

    def create_violin_plots(self, user_input: np.ndarray) -> plt.Figure:
        data = self._analyzer._data_provider.get_data()
        top_indices = self._analyzer.get_top_features_indices(n=2)
        feature_names = self._analyzer.get_feature_names()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Распределение признаков (Violin Plot)')

        for idx, (ax, feature_idx) in enumerate(zip(axes, top_indices)):
            feat_name = feature_names[feature_idx]
            feature_data = data[feat_name]
            user_val = user_input[feature_idx]

            parts = ax.violinplot(feature_data, showmeans=True, showmedians=True)
            self._style_violin(parts) # Вынесли стилизацию в метод
            
            # Точка пользователя
            ax.scatter(1, user_val, color='red', s=100, zorder=3, edgecolors='black')
            
            # Аннотация
            ax.annotate(f'Val: {user_val:.2f}', xy=(1, user_val), xytext=(20, 0),
                        textcoords='offset points', arrowprops=dict(arrowstyle='->'))

            ax.set_title(f"Feature: {feat_name}")
            ax.set_xticks([])
            
        plt.tight_layout()
        return fig

    def _style_violin(self, parts):
        """Вспомогательный метод для стилизации графика."""
        for body in parts['bodies']:
            body.set_facecolor('lightblue')
            body.set_alpha(0.7)


# Бизнес логика для объединения всех классов

class RegressionService:
    """
    Фасад, объединяющий все компоненты для удобного использования.
    """
    def __init__(self, model: Any, csv_path: str = 'lin_regres.csv'):
        self._data_provider = CsvTrainingDataProvider(csv_path)
        self._analyzer = ModelAnalyzer(model, self._data_provider)
        self._visualizer = RegressionVisualizer(self._analyzer)

    def process_prediction(self, features: List[float]) -> Tuple[str, str, Any, Any]:
        """
        Главный метод: делает предсказание, собирает статистику и рисует графики.
        """
        if self._analyzer._model is None:
            raise ValueError("Модель не загружена")

        input_array = np.array(features)
        
        # Предсказание
        # Reshape нужен, так как predict ждет 2D массив
        prediction = self._analyzer._model.predict(input_array.reshape(1, -1))[0]
        
        # Формула
        coefs_full = self._analyzer.get_coefficients_array()
        formula = FormulaFormatter.format_linear_equation(coefs_full[1:], coefs_full[0]) if coefs_full is not None else ""
        
        # Графики
        fig_scatter = self._visualizer.create_scatter_plots(input_array, prediction)
        fig_violin = self._visualizer.create_violin_plots(input_array)
        
        return f"{prediction:.5f}", formula, fig_scatter, fig_violin

    def get_model_quality(self) -> str:
        """Возвращает R2 score."""
        return self._analyzer.get_r2_score()