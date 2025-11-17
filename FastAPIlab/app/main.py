__author__ = "Kuvykin Nikita"



from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json
import os
import uvicorn
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.router_v1 import router_v1


app = FastAPI(
    title= "Линейная регрессия"
)

@app.get("/")
async def root():
    return{"message: xdd"}
    

app.include_router(router_v1)

