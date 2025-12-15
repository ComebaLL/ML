__author__ = "Kuvykin Nikita"

import uvicorn
import os
from fastapi import FastAPI
from pathlib import Path
from app.module_download import EnvConfigProvider, PickleModelLoader
from app.model_unit import RegressionService
from app.router_v1 import InferenceRouterV1, MetricsService


def create_application() -> FastAPI:
    """
    Фабрика приложения, сборка всех зависимостей.
    """
    print("запуск приложения")

    try:

        # Настройка конфига
        config = EnvConfigProvider(env_path='app/model.env')
        
        # Загрузка модели
        loader = PickleModelLoader()
        model_path = config.get_model_path()
        loaded_model = loader.load(model_path)
        print(f"Модель успешно загружена: {model_path}")

    except Exception as e:
        print(f"ошибка загрузки модели: {e}")
        loaded_model = None


    # Бизнес логика
    
    # Пути к вспомогательным файлам
    base_dir = Path(__file__).parent.parent 
    csv_path = base_dir / "lin_regres.csv"
    metrics_path = base_dir / "model" / "model_onfo.json"

    # Создаем сервисы и внедряем в них модель и пути
    regression_service = RegressionService(
        model=loaded_model, 
        csv_path=str(csv_path)
    )
    
    metrics_service = MetricsService(
        json_path=str(metrics_path)
    )


    
    # Создаем роутер и внедряем в него сервисы 
    api_router = InferenceRouterV1(
        model_service=regression_service, 
        metrics_service=metrics_service
    )


    # Сборка FastAPI 
    app = FastAPI(
        title="Линейная Регрессия API",
        version="1.0.0"
    )

    # Подключаем роутер
    app.include_router(api_router.router)

    # Корневой эндпоинт
    @app.get("/")
    async def root():
        return {
            "message": "API is running", 
            "docs_url": "/docs",
            "model_status": "loaded" if loaded_model else "failed"
        }

    return app


# Создаем экземпляр приложения
app = create_application()


if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(app, host="127.0.0.1", port=8000)