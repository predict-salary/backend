from typing import Literal, Union


class Config:
    bert_name: str = "cointegrated/rubert-tiny2"
    embedder_path: str = "./models/weight_store/model_best_macro_0.5759.pt"
    tabular_class_path: str = "./models/weight_store"
    tabular_model_path: str = "./models/weight_store/"
    chunk_size: int = 1024
    target_mapper_path: str = "./models/pipeline/preprocessing/mapper/target_mapper.pkl"
    vacancy_score_mapper_path: str = "./models/pipeline/preprocessing/mapper/name2score.pkl"
    device: Literal["cpu", "cuda:0", "auto"] = "cpu"
