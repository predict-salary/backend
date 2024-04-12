from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from models.main import ml_models, router as model_router
from models.pipeline import PipelinePredictSalary
from models.pipeline import Config


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device(Config.device) if Config.device != "auto" else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ml_models["pipeline"] = PipelinePredictSalary(device, Config.target_mapper_path)
    await ml_models["pipeline"].load_dependency()
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def home():
    return {"message": "predict salary by hh"}

app.include_router(model_router)
