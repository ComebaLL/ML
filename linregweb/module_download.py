__author__ = "Kuvykin Nikita"

import pickle
import os
from dotenv import load_dotenv

def load_model():
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

# Использование:
try:
    len_reg = load_model()
    print("Модель готова к использованию!")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")