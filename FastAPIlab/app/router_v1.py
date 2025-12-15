__author__ = "Kuvykin Nikita"

import os
import json
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from app.model_unit import RegressionService

# Вспомогательный класс для метрик 
class MetricsService:
    """
    Отвечает исключительно за получение данных о метриках.
    """
    def __init__(self, json_path: str):
        self._json_path = json_path

    def get_metrics(self) -> Dict[str, Any]:
        """Читает метрики из JSON файла."""
        if not os.path.exists(self._json_path):
            raise FileNotFoundError(f"Файл метрик не найден: {self._json_path}")
        
        try:
            with open(self._json_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError("Ошибка формата JSON в файле метрик")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка чтения метрик: {e}")


# Класс Роутера V1 
class InferenceRouterV1:
    """
    Класс, объединяющий все эндпоинты API версии 1.
    """
    def __init__(self, model_service: RegressionService, metrics_service: MetricsService):
        self._model_service = model_service
        self._metrics_service = metrics_service
        
        # Инициализируем FastAPI router
        self.router = APIRouter(prefix="/api/v1", tags=["v1"])
        
        # Регистрируем маршруты
        self._register_routes()

    def _register_routes(self):
        """Привязываем методы класса к путям URL."""
        self.router.add_api_route("/ping", self.pong, methods=["GET"])
        self.router.add_api_route("/predict", self.predict_handler, methods=["GET"])
        self.router.add_api_route("/metrics", self.metrics_handler, methods=["GET"])

    async def pong(self):
        """Проверка работоспособности."""
        return {"status": "ok"}

    async def predict_handler(self, x1: float, x2: float, x3: float, x4: float):
        """
        Эндпоинт для получения предсказания.
        """
        # Проверка на то, инициализирован ли сервис
        if not self._model_service:
             raise HTTPException(status_code=503, detail="Service Unavailable: Model not loaded")

        try:
            features = [x1, x2, x3, x4]
            result_text, _, _, _ = self._model_service.process_prediction(features)
            
            predict_value = float(result_text)

            return {
                "prediction": predict_value,
                "features": {
                    "x1": x1, "x2": x2, "x3": x3, "x4": x4,
                }
            }
        except Exception as e:
            # Важно возвращать 500 только при реальных крашах
            raise HTTPException(status_code=500, detail=f"Internal Calculation Error: {str(e)}")

    async def metrics_handler(self):
        """
        Эндпоинт для получения данных модели.
        """
        try:
            data = self._metrics_service.get_metrics()
            return data
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Metrics file not found on server")
        except ValueError:
            raise HTTPException(status_code=500, detail="Metrics file is corrupted")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")