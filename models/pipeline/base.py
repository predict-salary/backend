from abc import ABC, abstractmethod
from typing import Any


class PipelineTemplate(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    async def _preprocessing(self) -> Any:
        pass

    @abstractmethod
    async def prepare_result(self) -> Any:
        pass

    @abstractmethod
    async def load_dependency(self) -> Any:
        pass

    @abstractmethod
    async def __call__(self) -> Any:
        pass 
