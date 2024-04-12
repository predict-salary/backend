from typing import Optional

from pydantic import BaseModel


class BaseInference(BaseModel):
    ...


class VacancyInference(BaseInference):
    url: str


class VacancyResponse(BaseModel):
    salary_from: Optional[int]
    salary_to: Optional[int]
