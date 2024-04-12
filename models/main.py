from fastapi import APIRouter

from .models import VacancyInference, VacancyResponse


ml_models = {}
router = APIRouter(tags=["inference"])


@router.post("/api/v1/inference")
async def inference(data: VacancyInference) -> VacancyResponse:
    response = await ml_models["pipeline"](data.url)
    
    return VacancyResponse(salary_from=response[0], salary_to=response[1])
