# %%
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import cv2
import json
from torchvision import transforms
from PIL import Image, ImageOps
import math

from torch.optim import SGD, Adam, lr_scheduler, RMSprop

from os import path
import os
from typing import List, Dict

import erfnet

# %%


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
            (-1, 1, 2)).astype(np.int32)], False, idx + 1, 2)
    return label_img


class TensorConverter(object):
    def __init__(self, transform=None) -> None:
        self.transform = transform
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.scale = 0.8

    def __call__(self, img, points):
        if self.transform is not None:
            img, points = self.transform(img, points)
        h, w, _ = img.shape
        dst_mask = convertPointsToMask(
            int(h * self.scale / 8), int(w * self.scale / 8), [p * (self.scale / 8) for p in points])
        dst_mask = np.where(dst_mask, 1, 0)
        dst_mask = torch.from_numpy(dst_mask)
        img = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
        dst_img = self.toTensor(img)
        dst_img = self.normalize(dst_img)
        return dst_img, dst_mask


# %%

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


class ErfnetModule(pl.LightningModule):
    def __init__(self, num_classes, num_epochs, *args, **kwargs) -> None:
        super().__init__()
        self.num_epochs = num_epochs
        self.save_hyperparameters('num_epochs', 'num_classes')

        self.model = erfnet.Net(num_classes)

        weight = torch.Tensor([0.5, 1.])
        self.criterion = CrossEntropyLoss2d(weight)

    def forward(self, x):
        return self.model(x, only_encode=True)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), 5e-4,
                         (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

        def lambda1(epoch): return pow((1-((epoch-1)/self.num_epochs)), 0.9)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log('loss/train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        with torch.no_grad():
            output = self(data)
            loss = self.criterion(output, target)
            pred = output.argmax(dim=1)
            acc = (target == pred).sum() / float(torch.numel(target))
        self.log('loss/val', loss)
        self.log('metric/acc', acc)

# %%


class TusimpleDataModule(pl.LightningDataModule):
    def __init__(self, root_path, anno_path_list, num_workers=8, batch_size=6, *args, **kw_args) -> None:
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size

        anno_list = TusimpleDataset.getAnnotationList(
            root_path, anno_path_list)
        num_sample = int(0.9 * len(anno_list))
        train_anno, val_anno = torch.utils.data.random_split(
            anno_list, [num_sample, len(anno_list) - num_sample])
        self.train_ds = TusimpleDataset(root_path, train_anno,
                                        TensorConverter(RandomHomograpy(0.05, 10, 100, 50)))
        self.val_ds = TusimpleDataset(root_path, val_anno, TensorConverter())

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self.train_ds, num_workers=self.num_workers,
                                           batch_size=self.batch_size)

    def val_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self.val_ds, num_workers=self.num_workers,
                                           batch_size=self.batch_size)


# %%
root_path = '/home/bughunter/Downloads/tusimple'
anno_path_list = ['label_data_0531.json',
                  'label_data_0601.json', 'label_data_0313.json']

data_module = TusimpleDataModule(root_path, anno_path_list)
pl_model = ErfnetModule(2, 32)
trainer = pl.Trainer(gpus=1)
# trainer = pl.Trainer(gpus=1, resume_from_checkpoint='/home/bughunter/code/lane_test/lightning_logs/version_6/checkpoints/epoch=1.ckpt')

# %%
trainer.fit(pl_model, data_module)

# %%
for img, labels in data_module.val_dataloader():
    with torch.no_grad():
        output = pl_model(img)
        pred = output.argmax(dim=1)
    break

# %%
frame_id = 2
fig, (ax0, ax1, ax2) = plt.subplots(3)
ax0.imshow(img[frame_id].sum(axis=0))
ax1.imshow(labels[frame_id])
ax2.imshow(pred[frame_id])

# %%
