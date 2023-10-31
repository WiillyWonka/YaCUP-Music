import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import BertForSequenceClassification

class BaselineModel(nn.Module):
    def __init__(
        self,
        num_classes = 256,
        input_dim = 768,
        hidden_dim = 512
    ):
        super().__init__()
        self.num_classes = num_classes
        self.bn = nn.LayerNorm(hidden_dim)
        self.projector =  nn.Linear(input_dim, hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        

    def forward(self, embeds):
        x = [self.projector(x) for x in embeds]
        x = [v.mean(0).unsqueeze(0) for v in x]
        x = self.bn(torch.cat(x, dim = 0))
        x = self.lin(x)
        outs = self.fc(x)
        return outs

class BERT(nn.Module):
    def __init__(
        self,
        num_classes = 256
    ):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes, problem_type="multi_label_classification")

    def forward(self, embeds):
        out = self.model(inputs_embeds=embeds)
        return out.logits