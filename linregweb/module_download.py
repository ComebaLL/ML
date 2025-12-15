__author__ = "Kuvykin Nikita"

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any
from dotenv import load_dotenv

# Абстракция интерфейса
# ABC - Abstract Base class - превратит класс в щаблон

class IConfigProvider(ABC):
    """
    Интерфейс для получения конфигурации.
    Позволяет подменить источник настроек (файл .env, переменные среды Docker, конфиг в JSON).
    """
    @abstractmethod
    def get_model_path(self) -> str:
        """Возвращает путь к файлу модели."""
        pass


class IModelLoader(ABC):
    """
    Интерфейс для загрузчика модели.
    Позволяет подменить механизм загрузки (pickle, torch, joblib, onnx).
    """
    @abstractmethod
    def load(self, path: str) -> Any:
        """Загружает модель по указанному пути."""
        pass


# Загрузка модели
# наследование
class EnvConfigProvider(IConfigProvider):
    """
    Получает настройки из .env файла и переменных окружения.
    """
    def __init__(self, env_path: str = 'model.env'):
        # Загружаем переменные из указанного файла в окружение
        load_dotenv(env_path)

    def get_model_path(self) -> str:
        model_filename = os.getenv('MODULE_FILE_NAME')
        
        if not model_filename:
            raise ValueError("Переменная 'MODULE_FILE_NAME' не найдена в окружении.")
            
        return model_filename


class PickleModelLoader(IModelLoader):
    """
    Реализация загрузки через стандартный модуль pickle.
    """
    def load(self, path: str) -> Any:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден по пути: {path}")
            
        with open(path, 'rb') as file:
            return pickle.load(file)


# Бизнес логика загрузки модели

class ModelService:
    """
    Сервис, который оркестрирует процесс:
    Получает путь через ConfigProvider.
    Загружает файл через ModelLoader.
    """
    def __init__(self, config: IConfigProvider, loader: IModelLoader):
        self._config = config
        self._loader = loader

    def load_model(self) -> Any:
        """
        Основной метод для получения готовой модели.
        """
        path = self._config.get_model_path()
        model = self._loader.load(path)
        
        print(f"Модель успешно загружена: {path}")
        
        return model