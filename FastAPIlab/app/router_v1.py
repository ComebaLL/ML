__author__ = "Kuvykin Nikita"



import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json
import os
import uvicorn
import sys
from pathlib import Path
from app.model_unit import predict
from app.module_download import load_model

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    model = load_model()
    print("Модель загружена")
except Exception as e:
    print(f"Модель не загружена: {str(e)}")
    model = None

router_v1 = fastapi.APIRouter(prefix="/api/v1", tags=["v1"])

@router_v1.get("/ping")
async def pong():
    return {"status": "ok"}

@router_v1.get("/predict")
async def get_predict_model(x1: float, x2: float, x3: float, x4: float):
    """
    Энпоинт для получения  предсказания модели
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    try:
        result_text, _ = predict(model, x1, x2, x3, x4) # получаем предсказание

        predict_value = float(result_text)

        return{
            "prediction": predict_value,
            "features": {
                "x1": x1, "x2": x2, "x3": x3, "x4": x4,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")
    

@router_v1.get("/metrics")
async def get_metrics():
    """
    Эндпоинт для получения данных модели
    """

    try:

        # путь к json файлу
        json_path = os.path.join(os.path.dirname(__file__), "..", "model", "model_onfo.json") 
        json_path = os.path.abspath(json_path)

        if not os.path.exists(json_path):
            print("фалй не найден")
        else:
            with open(json_path, "r", encoding='utf-8') as f:
                metrics_data = json.load(f)
            print("метрики загружены")
        
        return JSONResponse(content=metrics_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Файле не найден")
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Ошибка чтения файла: {str(e)}")