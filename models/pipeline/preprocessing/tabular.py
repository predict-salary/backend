import re
import pickle
from itertools import permutations
from typing import Any, Dict

import torch

import pandas as pd
from pytorch_tabular.tabular_datamodule import TabularDataset
import warnings
warnings.filterwarnings("ignore")

from ..config import Config


class TabularProcessor:
    def __init__(self, tabular_class) -> None:
        self.tabular_class = tabular_class
        self.numeric_features = ["vacancy_name_score", "vacancy_name_len"]
        self.categorical_features = [
                                "work_schedule", "city", "accredit_it", "state",
                                "salary_currency", "salary_gross", "vacancy_experience", "vacancy_type", "created_quarter",
                                "created_month", "trusted"
        ]
        self.temp_columns = ["vacancy_name", "created_date"]
        with open(Config.vacancy_score_mapper_path, "rb") as f:
            self.name2score = pickle.load(f)

    def _features_collector(self, tabular_data: Dict[str, Any]):
        columns = self.numeric_features + self.categorical_features 
        template_tabular_sample = {col: [None] for col in columns}

        company = tabular_data["company"]
        template_tabular_sample["trusted"] = [company["@trusted"]]
        template_tabular_sample["state"] = [company["@state"]]
        template_tabular_sample["work_schedule"] = [tabular_data["@workSchedule"]]
        template_tabular_sample["accredit_it"] = [company["accreditedITEmployer"] if "accreditedITEmployer" in company.keys() else False]

        template_tabular_sample["city"] = [tabular_data["area"]["name"]]

        compensation = tabular_data["compensation"]
        template_tabular_sample["salary_currency"] = [compensation["currencyCode"] if "currencyCode" in compensation.keys() else None]
        template_tabular_sample["salary_gross"] = [compensation["gross"] if "gross" in compensation.keys() else False]

        template_tabular_sample["vacancy_experience"] = [tabular_data["workExperience"]]
        template_tabular_sample["vacancy_type"] = [tabular_data["type"]]

        template_tabular_sample["vacancy_name"] = [tabular_data["name"]]
        template_tabular_sample["created_date"] = [tabular_data["publicationDate"]]

        return template_tabular_sample

    def __creator_score(self, row):
        vacancy_comb = list(map(lambda x: tuple(sorted(x)), list(permutations(row, 2))[:len(row) * 2 - 2]))
        score = [self.name2score.get(name, 0) for name in vacancy_comb]
        return sum(score) / len(score) if len(score) != 0 else 0
    
    def _mapper(self, data: Dict[str, Any]) -> None:
        r = re.compile("[а-яА-Яa-zA-Z]+")
        vacancy_name = r.findall(data["vacancy_name"][0].lower())
        data["vacancy_name_score"] = [self.__creator_score(vacancy_name)]
        data["vacancy_name_len"] = [len(data["vacancy_name"][0])]
    
    def preprocessor(self, data: Dict[str, Any]):
        data = self._features_collector(data)
        self._mapper(data)
        data: pd.DataFrame = pd.DataFrame(data)
        data["created_date"] = pd.to_datetime(data["created_date"], format="mixed")
        data["created_quarter"] = data["created_date"].dt.quarter
        data["created_month"] = data["created_date"].dt.month

        data.drop(self.temp_columns, axis=1, inplace=True)
        # print(data.T)

        # tabular_dataset = TabularDataset(
        #     task=self.tabular_class.datamodule.config.task,
        #     data=self.tabular_class.datamodule._prepare_inference_data(data),
        #     categorical_cols=self.tabular_class.datamodule.config.categorical_cols,
        #     continuous_cols=self.tabular_class.datamodule.config.continuous_cols,
        # )[0]

        # for key in tabular_dataset.keys():
        #     tabular_dataset[key] = torch.from_numpy(tabular_dataset[key])[None, :]
        
        return data

    def __call__(self, tabular_data: Dict[str, Any]) -> Any:
        data = self.preprocessor(tabular_data)
        return data
