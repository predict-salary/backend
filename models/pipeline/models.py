from typing import Dict
import pickle

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from pytorch_tabular import TabularModel

from .config import Config


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class BertEmbedder(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

        self.seq_0 = nn.Sequential(
            nn.Linear(312, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
    def forward(self, inputs):
        x = self.bert(**inputs)
        x = (x["last_hidden_state"] * inputs["attention_mask"][:, :, None]).sum(dim=1) / inputs["attention_mask"][:, :, None].sum(dim=1)
        x = self.seq_0(torch.mean(x, dim=0))

        return x


class BertClassification(nn.Module):
    def __init__(self, embedder, tabular_model):
        super().__init__()
        self.embedder = embedder
        self.tabular_model = tabular_model #torch.load(tabular_model)

        self.seq_0 = nn.Sequential(
            nn.Linear(344, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
    def forward(self, inputs, tabular_data):
        tabular_x = self.tabular_model.predict(tabular_data, ret_logits=True)
        tab_emb = torch.from_numpy(tabular_x.iloc[:, -32:].to_numpy())
        x = self.embedder(inputs)
        x = torch.cat((x[None, :], tab_emb), dim=1)
        x = self.seq_0(x)

        return x


async def load_model():
    bert_model =  AutoModel.from_pretrained(Config.bert_name)

    embedder = BertEmbedder(bert_model)
    # embedder.load_state_dict(torch.load(Config.embedder_path))
    embedder.seq_0 = Identity()

    with open(f"{Config.tabular_class_path}/tabular_class.pkl", "rb") as f:
        tabular_class = pickle.load(f)
    #     # tabular_model.load_state_dict(f"{Config.tabular_model_path}/tabular_tf.pt")
    tabular_model = TabularModel.load_model(Config.tabular_model_path, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(Config.bert_name)
    model = BertClassification(embedder, tabular_model)
    model.load_state_dict(torch.load(Config.embedder_path), strict=False)
    model.eval()

    return model, tokenizer, tabular_class
