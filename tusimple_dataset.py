import json
from typing import List, Dict
from os import path

import numpy as np
import torch
import cv2
from torchvision import transforms


class TusimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str, anno_list: List, transform=None):
        super().__init__()
        self.transform = transform
        self.root_path = root_path
        self.anno_list = anno_list

    @staticmethod
    def getAnnotationList(root_path, anno_path_list):
        anno_list = []
        for anno_path in anno_path_list:
            with open(path.join(root_path, anno_path), 'r') as f:
                anno_list += f.readlines()
        return anno_list

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, idx):
        anno_dict = json.loads(self.anno_list[idx])
        img = cv2.imread(path.join(self.root_path, anno_dict['raw_file']))
        # img = Image.open(path.join(self.root_path, anno_dict['raw_file']))
        assert img is not None

        lanes = np.array(anno_dict['lanes'])
        h_samples = np.array(anno_dict['h_samples'])
        def filter_lane(lane): return lane[lane[:, 0] >= 0]
        points = [filter_lane(np.stack((lane, h_samples), axis=1))
                  for lane in lanes]
        points = [p for p in points if p.size > 0]

        if self.transform is not None:
            return self.transform(img, points)
        return img, points


class RandomHomograpy(object):
    def __init__(self, scale, rot, x, y) -> None:
        def transform(img, points):
            h, w, _ = img.shape
            rand_scale = np.random.uniform(1 - scale, 1 + scale)
            rand_rot = np.random.uniform(-rot, rot)
            rand_x = np.random.uniform(-x, x)
            rand_y = np.random.uniform(-y, y)
            mat = np.concatenate((cv2.getRotationMatrix2D(
                (w / 2, h / 2), rand_rot, rand_scale), [[0, 0, 1]]))
            mat[0, 2] += rand_x
            mat[1, 2] += rand_y
            dst_img = cv2.warpPerspective(img, mat, (w, h))
            dst_points = [cv2.perspectiveTransform(
                p.reshape(1, -1, 2).astype(float), mat).reshape(-1, 2) for p in points]
            return dst_img, dst_points
        self.tranform = transform

    def __call__(self, img, points):
        return self.tranform(img, points)


def convertPointsToMask(height: int, width: int, points: List[np.ndarray]):
    label_img = np.zeros((height, width), dtype=np.uint8)
    for idx, point in enumerate(points):
        cv2.polylines(label_img, [point.reshape(
            (-1, 1, 2)).astype(np.int32)], False, idx + 1, 18)
    return label_img


class TensorConverter(object):
    def __init__(self, transform=None) -> None:
        self.transform = transform
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.dst_w = 800
        self.dst_h = 288

        w, h = 1280, 720
        upper_scale = w / 6
        upper_line = 220
        lower_scale = w / 2 * 2
        lower_line = h
        hws = w / 2
        ps = np.float32([[hws - upper_scale, upper_line], [hws + upper_scale, upper_line],
                        [hws - lower_scale, lower_line], [hws + lower_scale, lower_line]])
        pd = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        self.trans_mat = cv2.getPerspectiveTransform(ps, pd)

    def __call__(self, img, points):
        h, w, _ = img.shape
        img = cv2.warpPerspective(img, self.trans_mat, (w, h))
        points = [cv2.perspectiveTransform(
            p.reshape(1, -1, 2).astype(float), self.trans_mat).reshape(-1, 2) for p in points]

        if self.transform is not None:
            img, points = self.transform(img, points)
        dst_mask = convertPointsToMask(h, w, [p for p in points])

        dst_mask = cv2.resize(dst_mask, (int(self.dst_w / 8), int(self.dst_h / 8)), 0, 0, cv2.INTER_NEAREST)
        dst_mask = torch.from_numpy(dst_mask)
        img = cv2.resize(img, (self.dst_w, self.dst_h))
        dst_img = self.toTensor(img)
        dst_img = self.normalize(dst_img)
        return dst_img, dst_mask
