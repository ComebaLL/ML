__author__ = "Kuvykin Nikita"

import pickle
import os
import json
from dotenv import load_dotenv
from model_unit import create_violin_plots, create_scatter_plots,create_prediction_data
from typing import Any, Tuple

class LinnRegressModel:
    """
    Класс линейной регрессии
    """

    def __init__(self, model = None):
        """
        Конструктор
        """

        self._model = model
        self._metrics = {
            'r2' : [],
            'MSE' : [],
            'MAE' : []
        }
        # todo
        self._text = description    

    def load_model(self) -> None:
        """
        Загружает модель из файла, указанного в model.env
        """
        # Загружаем переменные из файла .env
        load_dotenv('model.env')
    
        # Получаем имя файла модели
        model_filename = os.getenv('MODULE_FILE_NAME')
    
        if not model_filename:
            raise ValueError("MODULE_FILE_NAME не найден в model.env")
    
        # Проверяем существование файла
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Файл модели '{model_filename}' не найден")
    
        # Загружаем модель
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
    
        print(f"Модель успешно загружена из файла: {model_filename}")
        return model
    

    def load_metrics(self, filepath: str = 'metrics_model.json') -> None:
        """
        Загрузка метрик из JSON файла.
        Метрики сохраняются как массивы.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_metrics = json.load(f)
            
            print(f"Загруженные метрики из {filepath}:")
            
            # Загружаем массив
            for metric_name in ['r2', 'MSE', 'MAE']:
                if metric_name in loaded_metrics:
                    value = loaded_metrics[metric_name]
                    
                    if isinstance(value, list):
                        self.metrics[metric_name] = value
                        print(f"  {metric_name}: массив из {len(value)} значений")
                    elif isinstance(value, (int, float)):
                        # Если значение одно, превращаем в список с одним элементом
                        self.metrics[metric_name] = [value]
                        print(f"  {metric_name}: единичное значение, преобразовано в список")
                    else:
                        print(f"  {metric_name}: неизвестный формат, оставлен пустым списком")
                        self.metrics[metric_name] = []
                else:
                    print(f"  Метрика '{metric_name}' не найдена в файле")
                    self.metrics[metric_name] = []
            
            print(f"\nМетрики успешно загружены из {filepath}")
            
        except FileNotFoundError:
            print(f"Файл {filepath} не найден")
        except json.JSONDecodeError:
            print(f"Ошибка при чтении JSON файла {filepath}")
        except Exception as e:
            print(f"Ошибка при загрузке метрик: {e}")
    


    def print_metrics(self) -> None:
        """
        Вывод метрик в консоль 
        """
        print("Метрики:")
        
        for metric_name, metric_array in self.metrics.items():
            print(f"\n{metric_name}:")
            
            if metric_array:
                # Выводим значения с форматированием
                for i, value in enumerate(metric_array, 1):
                    print(f"  [{i}] = {value}")
            else:
                print("  Массив пуст")

    
    def predict_model(self, x1: float, x2: float, x3: float, x4: float) -> Tuple[str, Any, Any]:
        """
        Функция для предсказания и создания графиков 
        """
        if self.model is None:
            return "Ошибка: Модель не загружена", None, None
        
        try:
            # Создаем DataFrame с правильными именами признаков
            input_data = self.create_prediction_data(x1, x2, x3, x4)
            
            # Делаем предсказание
            prediction = self.model.predict(input_data)
            
            # Создаем графики
            plot_obj = self.create_scatter_plots(x1, x2, x3, x4)
            plot_obj2 = self.create_violin_plots(x1, x2, x3, x4)
            
            # Возвращаем результат
            result_text = f"Предсказанное значение: {prediction[0]:.5f}"
            return result_text, plot_obj, plot_obj2
        
        except Exception as e:
            return f"Ошибка предсказания: {e}", None, None
        
    
    def simple_predict(self, x1: float, x2: float, x3: float, x4: float) -> float:
        """Простое предсказание"""
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        input_data = self.create_prediction_data(x1, x2, x3, x4)
        prediction = self.model.predict(input_data)
        return float(prediction[0])