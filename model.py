from typing import Dict
from dataclasses import dataclass, asdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


import erfnet


@dataclass
class Output:
    lane_segment: torch.Tensor
    embedding_vector: torch.Tensor = None

    def predictSegment(self) -> torch.Tensor:
        return self.lane_segment > 0


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = erfnet.Net(1)

    def forward(self, x):
        lane_segment=self.model(x, only_encode=True).squeeze(dim=1)
        return dict(lane_segment=lane_segment)


class SegmentationLoss(nn.Module):
    def __init__(self, pos_weight_val = 2.) -> None:
        super().__init__()
        pos_weight = torch.Tensor([pos_weight_val])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, target: torch.Tensor, lane_segment: torch.Tensor, *args, **kwargs):
        return self.criterion(lane_segment, (target > 0).float())
