from typing import Any
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import pandas as pd

def nll_loss(output, target):
    return F.nll_loss(output, target)

class WeightedBCE(torch.nn.Module):
    def __init__(self, class_count_path: str) -> None:
        super(WeightedBCE, self).__init__()

        class_count_df = pd.read_csv(class_count_path)

        class_count = torch.zeros(class_count_df.shape[0], requires_grad=False)

        for i in range(class_count_df.shape[0]):
            class_count[class_count_df.iloc[i]['tags']] = class_count_df.iloc[i]['count']

        self._weights = 1 / class_count
        self._weights /= self._weights.sum()

        self.bce = BCEWithLogitsLoss(reduction='none')

    def to(self, device):
        super().to(device)
        self._weights = self._weights.to(device)
        return self

    def __call__(self, predict, target) -> torch.tensor:
        loss = self.bce(predict, target)
        loss = (loss * self._weights).mean()
        return loss

        


