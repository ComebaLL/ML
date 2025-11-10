__author__ = "Kuvykin Nikita"



from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json
import os
import uvicorn
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from module_download import load_model
from model_unit import predict


try:
    model = load_model()
    print("Модель загружена")
except Exception as e:
    print(f"Модель не загружена: {str(e)}")
    model = None

app = FastAPI(
    title= "Линейная регрессия"
)

@app.get("/ping")
async def pong():
    return {"status": "ok"}

@app.get("/predict")
async def get_predict_model(x1: float, x2: float, x3: float, x4: float):
    """
    Энпоинт для получения  предсказания модели
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    try:
        result_text, _ = predict(model, x1, x2, x3, x4) # получаем предсказание

        predict_value = float(result_text.split(":")[1].strip())

        return{
            "prediction": predict_value,
            "features": {
                "x1": x1, "x2": x2, "x3": x3, "x4": x4,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")
    

@app.get("/metrics")
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
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

