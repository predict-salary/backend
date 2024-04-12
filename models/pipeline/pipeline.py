from typing import Any, Dict, Tuple
import json
import pickle

import aiohttp
import torch
from bs4 import BeautifulSoup

from .base import PipelineTemplate
from .models import load_model
from .preprocessing import TextPreprocessor, TabularProcessor


class PipelinePredictSalary(PipelineTemplate):
    def __init__(self, device, mapper_path: str) -> None:
        super().__init__()
        self.model = None
        self.text_prepocessor = None
        self.device = device
        with open(mapper_path, "rb") as f:
            self.target_mapper = pickle.load(f)

    async def _preprocessing(self, url: str) -> Tuple[Dict[str, torch.Tensor]]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                html_page = await response.text()

        soup = BeautifulSoup(html_page, "html.parser")
        
        data = json.loads(soup.find_all("template", attrs={"style": "display:none"})[-1].text)["vacancyView"]
        vacancy_description = BeautifulSoup(data["description"], "html.parser").text
        
        tokens = self.text_prepocessor(vacancy_description, self.device)
        tabular_data = self.tabular_preprocessor(data)

        return tokens, tabular_data
            
    async def prepare_result(self, result: int) -> Any:
        async def _salary_mapper(x: int):
            if x == 6:
                return (600_000, None)
            for (left_salary, right_salary), idx in self.target_mapper.items():
                if idx == x:
                    return left_salary, right_salary
            return None
        
        result = await _salary_mapper(result)
        if result is None:
            raise ValueError(f"Class {result} not find in mapper!")
        
        return result

    async def load_dependency(self) -> Any:
        self.model, tokenizer, tabular_class = await load_model()
        self.model = self.model.to(self.device)
        self.text_prepocessor = TextPreprocessor(tokenizer)
        self.tabular_preprocessor = TabularProcessor(tabular_class)
        
    async def __call__(self, url: str) -> Any:
        vacancy_tokens, tabular_data = await self._preprocessing(url)

        output = self.model(vacancy_tokens, tabular_data)
        output = abs(output)
        output = torch.nn.functional.softmax(output.detach(), dim=-1)

        result = torch.argmax(output, dim=-1).cpu().item()
    
        if result == 6:
            return (None, None)

        return await self.prepare_result(result)
