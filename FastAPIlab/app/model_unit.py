__author__ = "Kuvykin Nikita"


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_training_data():
    """
    Загружает данные из lin_regres.csv
    """
    try:
        data = pd.read_csv('lin_regres.csv')
        return data
    except FileNotFoundError:
        print("Файл lin_regres.csv не найден")
        return None
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None


def get_r2_score(model):
    """
    Вычисляет R2 score на исходных данных из lin_regres.csv
    """
    if model is None or not hasattr(model, 'score'):
        return "Не доступно"
    
    try:
        # Загружаем данные
        data = load_training_data()
        if data is None:
            return "Данные не найдены"
        
        # Разделяем на признаки и целевую переменную (уже с правильными именами)
        X_data = data[['x1', 'x2', 'x3', 'x4']]
        y_data = data['y']
        
        # Вычисляем R2 score
        r2 = model.score(X_data, y_data)
        return f"{r2:.4f}"
    
    except Exception as e:
        print(f"Ошибка вычисления R2: {e}")
        return "Ошибка вычисления"
    


def get_model_coefficients(model):
    """
    Возвращает коэффициенты модели в виде numpy array
    """
    if model is None or not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
        return None
    
    coefficients = model.coef_
    intercept = model.intercept_
    
    # numpy array: [intercept, coef1, coef2, coef3, coef4]
    coef_array = np.concatenate([[intercept], coefficients])
    
    return coef_array


def get_top_features(model):
    """
    Определяет 2 наиболее важных признака на основе коэффициентов модели
    """
    if model is None or not hasattr(model, 'coef_'):
        return [0, 1]
    
    coefficients = model.coef_
    # Получаем индексы двух признаков с наибольшими абсолютными коэффициентами
    top_indices = np.argsort(np.abs(coefficients))[-2:][::-1]
    return top_indices.tolist()


def create_prediction_data(x1, x2, x3, x4):
    """
    Создает DataFrame для предсказания
    """
    return pd.DataFrame({
        'x1': [x1],
        'x2': [x2],
        'x3': [x3],
        'x4': [x4]
    })


def create_scatter_plots(model, x1, x2, x3, x4):
    """
    Создание диаграмм рассеяния для топ-2 важных признаков
    """
    data = load_training_data()
    if data is None or model is None:
        return None
    
    # Получаем наиболее важные признаки (топ-2)
    top_features_idx = get_top_features(model)
    feature_names = ['x1', 'x2', 'x3', 'x4']
    top_features = [feature_names[i] for i in top_features_idx]
    
    # Входные данные пользователя
    user_input = np.array([x1, x2, x3, x4])
    prediction = model.predict([user_input])[0]
    
    # Создаем графики - 1 строка, 2 столбца
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Диаграммы рассеяния: Важнейшие признаки vs Целевая переменная')
    
    for idx, (ax, feature_idx, feature_name) in enumerate(zip(axes, top_features_idx, top_features)):
        # Создаем DataFrame для текущего признака
        plot_data = pd.DataFrame({
            'x': data[feature_name], 
            'y': data['y']
        })
        
        # Отображение точек обучающей выборки
        sns.scatterplot(data=plot_data, x='x', y='y', 
                       alpha=0.5, label='Обучающие данные', ax=ax)
        
        # Добавление линии регрессии
        sns.regplot(data=plot_data, x='x', y='y', 
                   scatter=False, color='blue', 
                   line_kws={'label': 'Линия регрессии'}, ax=ax)
        
        # Отображение точки введенных пользователем данных
        ax.scatter(user_input[feature_idx], prediction, 
                  color='red', s=100, label='Ваши данные', edgecolors='black')
        
        ax.set_xlabel(f'Признак {feature_name}')
        ax.set_ylabel('Целевая переменная (y)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def predict(model, x1, x2, x3, x4):
    """
    Функция для предсказания и создания графиков
    """
    if model is None:
        return "Ошибка: Модель не загружена", None
    
    try:
        # Создаем DataFrame с правильными именами признаков
        input_data = create_prediction_data(x1, x2, x3, x4)
        
        # Делаем предсказание
        prediction = model.predict(input_data)
        
        # Создаем графики
        plot_obj = create_scatter_plots(model, x1, x2, x3, x4)
        
        # Возвращаем результат
        result_text = f"{prediction[0]:.5f}"
        return result_text, plot_obj
    
    except Exception as e:
        return f"Ошибка предсказания: {e}", None