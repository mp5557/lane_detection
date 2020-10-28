from typing import Dict
from dataclasses import dataclass, asdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


import erfnet
from erfnet import non_bottleneck_1d


@dataclass
class Output:
    lane_segment: torch.Tensor
    embedding_vector: torch.Tensor = None

    def predictSegment(self) -> torch.Tensor:
        return self.lane_segment > 0


class Head(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.model = nn.Sequential(
            non_bottleneck_1d(input_dim, 0.3, 2),
            non_bottleneck_1d(input_dim, 0.3, 4),
            nn.Conv2d(
                input_dim, output_dim, 1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = erfnet.Encoder()
        self.seg_head = Head(self.backbone.dim, 1)

    def forward(self, x):
        feature = self.backbone(x)
        lane_segment = self.seg_head(feature).squeeze(dim=1)
        return dict(lane_segment=lane_segment)


class SegmentationLoss(nn.Module):
    def __init__(self, pos_weight_val=2.) -> None:
        super().__init__()
        pos_weight = torch.Tensor([pos_weight_val])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, target: torch.Tensor, lane_segment: torch.Tensor, *args, **kwargs):
        return self.criterion(lane_segment, (target > 0).float())
