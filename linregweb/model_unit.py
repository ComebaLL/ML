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
    
    # [intercept, coef1, coef2, coef3, coef4]
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
    return fig


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
        plot_obj2 = create_violin_plots(model, x1, x2, x3, x4)
        
        # Возвращаем результат
        result_text = f"{prediction[0]:.5f}"
        return result_text, plot_obj,plot_obj2
    
    except Exception as e:
        return f"Ошибка предсказания: {e}", None
    

def coefficients_for_formul(coef_array):
    """
    Преобразует массив коэффициентов в красивую LaTeX формулу
    """
    if coef_array is None or len(coef_array) != 5:
        return "y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\beta_3 x_3 + \\beta_4 x_4"
    
    intercept = coef_array[0]
    coefficients = coef_array[1:]
    
    # Создаем уравнение LaTeX
    equation_parts = []
    
    # Свободный член
    if abs(intercept) > 0.001: 
        equation_parts.append(f"{intercept:.4f}")
    
    # Коэффициенты с переменными
    for i, coef in enumerate(coefficients):
        if abs(coef) > 0.001: 
            sign = " + " if coef >= 0 else " - "
            abs_coef = abs(coef)
            equation_parts.append(f"{sign}{abs_coef:.4f} x_{{{i+1}}}")
    
    # Собираем уравнение
    if equation_parts:
        if equation_parts[0].startswith(' + '):
            equation_parts[0] = equation_parts[0][3:]
        equation = "y = " + "".join(equation_parts)
    else:
        equation = "y = 0"
    
    return equation


def create_violin_plots(model, x1, x2, x3, x4):
    """
    Создание скрипичных диаграмм для топ-2 важных признаков
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
    fig.suptitle('Скрипичные диаграммы: Распределение важнейших признаков')
    
    for idx, (ax, feature_idx, feature_name) in enumerate(zip(axes, top_features_idx, top_features)):
        # Создаем данные для скрипичной диаграммы
        feature_data = data[feature_name]
        user_value = user_input[feature_idx]
        
        # Создаем скрипичную диаграмму
        violin_parts = ax.violinplot(feature_data, positions=[0], 
                                   showmeans=True, showmedians=True)
        
        # Настраиваем внешний вид скрипичной диаграммы
        violin_parts['bodies'][0].set_facecolor('lightblue')
        violin_parts['bodies'][0].set_alpha(0.7)
        violin_parts['cmins'].set_color('darkblue')
        violin_parts['cmaxes'].set_color('darkblue')
        violin_parts['cbars'].set_color('darkblue')
        violin_parts['cmeans'].set_color('red')
        violin_parts['cmedians'].set_color('green')
        
        # Добавляем точку пользовательских данных
        ax.scatter(0, user_value, color='red', s=100, 
                  label='Ваше значение', edgecolors='black', zorder=3)
        
        # Добавляем аннотацию для пользовательского значения
        ax.annotate(f'Ваше значение: {user_value:.2f}', 
                   xy=(0, user_value), xytext=(10, 10),
                   textcoords='offset points', ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Настройка осей и легенды
        ax.set_ylabel(f'Значение признака {feature_name}')
        ax.set_xlabel('Распределение признака')
        ax.set_xticks([0])
        ax.set_xticklabels([f'Признак {feature_name}'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем статистическую информацию
        stats_text = f'Медиана: {feature_data.median():.2f}\nСреднее: {feature_data.mean():.2f}\nСтд: {feature_data.std():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt