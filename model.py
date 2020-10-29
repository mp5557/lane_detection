from typing import Dict
from dataclasses import dataclass, asdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.functional import embedding


import erfnet
from erfnet import non_bottleneck_1d


# @dataclass
# class Output:
#     lane_segment: torch.Tensor
#     embedding_vector: torch.Tensor = None

#     def predictSegment(self) -> torch.Tensor:
#         return self.lane_segment > 0

def predictSegment(lane_segment: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return lane_segment > 0


def visualizeEmbedding(targets: torch.Tensor, embedding_vector: torch.Tensor, *args, **kwargs):
    max_lane_count = 5
    img_list = []
    for idx in range(1, max_lane_count + 1):
        vs, us = torch.nonzero(targets == idx, as_tuple=True)
        if vs.numel() > 0:
            mean_vec = embedding_vector[:, vs, us].mean(dim=1)
            diff = embedding_vector - mean_vec.reshape((-1, 1, 1))
            img_list.append(torch.sum(diff ** 2, dim=0))
    return img_list


class Head(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.model = nn.Sequential(
            non_bottleneck_1d(input_dim, 0.3, 2),
            nn.Conv2d(
                input_dim, output_dim, 1, stride=1, padding=0, bias=True))

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    def __init__(self, embedding_dim=4) -> None:
        super().__init__()
        self.backbone = erfnet.Encoder()
        self.seg_head = Head(self.backbone.dim, 1)
        self.embedding_head = Head(self.backbone.dim, embedding_dim)

    def forward(self, x):
        feature = self.backbone(x)
        with torch.no_grad():
            assert torch.all(torch.isfinite(x))
            assert torch.all(torch.isfinite(feature))
        lane_segment = self.seg_head(feature).squeeze(dim=1)
        embedding_vector = self.embedding_head(feature)
        return dict(lane_segment=lane_segment, embedding_vector=embedding_vector)
        # return dict(lane_segment=lane_segment)


class SegmentationLoss(nn.Module):
    def __init__(self, pos_weight_val=2.) -> None:
        super().__init__()
        pos_weight = torch.Tensor([pos_weight_val])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, target: torch.Tensor, lane_segment: torch.Tensor, *args, **kwargs):
        return self.criterion(lane_segment, (target > 0).float())


class ClusteringLoss(nn.Module):
    def __init__(self, max_lane_count=5, delta_v=1., delta_d=6.) -> None:
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.max_lane_count = max_lane_count
        self.register_buffer('zero', torch.Tensor([0.]))

    def forward(self, targets: torch.Tensor, embedding_vector: torch.Tensor, *args, **kwargs):
        with torch.no_grad():
            assert torch.all(torch.isfinite(embedding_vector))

        point_count = 0
        tot_dist = self.zero
        tot_var = self.zero
        b_dim = targets.shape[0]

        for bid in range(b_dim):
            mean_vec_list = []
            for idx in range(1, self.max_lane_count + 1):
                vs, us = torch.nonzero(targets[bid] == idx, as_tuple=True)
                if vs.numel() > 1:
                    mean_vec = embedding_vector[bid, :, vs, us].mean(dim=1)
                    diff = embedding_vector[bid, :, vs, us] - \
                        mean_vec.reshape((-1, 1))
                    # tot_dist = tot_dist + \
                    #     torch.sum(
                    #         F.relu(torch.sqrt(torch.sum(diff ** 2, dim=0)) - self.delta_v) ** 2)
                    tot_dist = tot_dist + \
                        torch.sum(
                            F.relu(torch.linalg.norm(diff, dim=0) - self.delta_v) ** 2)
                    point_count = point_count + vs.numel()
                    mean_vec_list.append(mean_vec)
            if len(mean_vec_list) > 1:
                mean_vec_tensor = torch.stack(mean_vec_list)
                tot_var = tot_var + \
                    torch.mean(
                        F.relu(self.delta_d - F.pdist(mean_vec_tensor)) ** 2)
        tot_dist = tot_dist / point_count if point_count > 0 else self.zero

        assert torch.isfinite(tot_dist)
        assert torch.isfinite(tot_var)

        return tot_dist + tot_var / b_dim
