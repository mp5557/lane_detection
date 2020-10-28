# %%
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from torch.optim import SGD, Adam, lr_scheduler, RMSprop
import torch.nn as nn

from os import path

import erfnet
from model import Output

from tusimple_dataset import *
import model
# %%


class ErfnetModule(pl.LightningModule):
    def __init__(self, num_classes, num_epochs, *args, **kwargs) -> None:
        super().__init__()
        self.num_epochs = num_epochs
        self.save_hyperparameters('num_epochs', 'num_classes')

        self.model = model.Model()

        self.criterion = model.SegmentationLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), 5e-4,
                         (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

        def lambda1(epoch): return pow((1-((epoch-1)/self.num_epochs)), 0.9)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        loss = self.criterion(target, **output)
        self.log('loss/train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        with torch.no_grad():
            output = self(data)
            loss = self.criterion(target, **output)

            pred = Output(**output).predictSegment()
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
root_path = '/home/dji/docker_share/dataset/tusimple/train_set'
anno_path_list = ['label_data_0531.json',
                  'label_data_0601.json', 'label_data_0313.json']

data_module = TusimpleDataModule(root_path, anno_path_list)
pl_model = ErfnetModule(1, 32)
trainer = pl.Trainer(gpus=1)
# trainer = pl.Trainer(
#     gpus=1, resume_from_checkpoint='/home/bughunter/code/lane_test/lightning_logs/version_6/checkpoints/epoch=1.ckpt')

# %%
trainer.fit(pl_model, data_module)

# %%
for img, labels in data_module.val_dataloader():
    with torch.no_grad():
        output = model.Output(**pl_model(img))
        pred = output.predictSegment()
    break

# %%
frame_id = 2
fig, (ax0, ax1, ax2) = plt.subplots(3)
ax0.imshow(img[frame_id].sum(axis=0))
ax1.imshow(labels[frame_id])
ax2.imshow(pred[frame_id])

# %%
