from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from ..config import Config


class TextPreprocessor:
    def __init__(self, tokenier: AutoTokenizer) -> None:
        self.tokenizer = tokenier

    @staticmethod
    def chunker(tokens: Dict[str, torch.Tensor], chunk_size: int = 512) -> List[str]:
        input_ids = list(tokens["input_ids"][0].split(chunk_size - 2))
        attention_mask = list(tokens["attention_mask"][0].split(chunk_size - 2))

        cls_token_id = 2
        eos_token_id = 3

        for i in range(len(input_ids)):
            input_ids[i] = torch.cat([torch.Tensor([cls_token_id]), input_ids[i], torch.Tensor([eos_token_id])])
            attention_mask[i] = torch.cat([torch.Tensor([1]), attention_mask[i], torch.Tensor([1])])

            pad_len = chunk_size - len(input_ids[i])
            if pad_len > 0:
                input_ids[i]= torch.cat([input_ids[i], torch.Tensor([0]*pad_len)])
                attention_mask[i] = torch.cat([attention_mask[i], torch.Tensor([0]*pad_len)])
                
        tokens["input_ids"] = torch.stack(input_ids)
        tokens["attention_mask"] = torch.stack(attention_mask)
            
        return tokens

    def __call__(self, text: str, device: torch.device) -> Any:
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)
        tokens = self.chunker(tokens, Config.chunk_size)
        tokens["input_ids"] = tokens["input_ids"].to(device).long()
        tokens["attention_mask"] = tokens["attention_mask"].to(device).int()

        return tokens
