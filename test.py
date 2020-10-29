# %%
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from torch.optim import SGD, Adam, lr_scheduler, RMSprop
import torch.nn as nn
import torch.nn.functional as F

from os import path

from model import predictSegment

from tusimple_dataset import *
import model
# %%


class ErfnetModule(pl.LightningModule):
    def __init__(self, num_classes, num_epochs, *args, **kwargs) -> None:
        super().__init__()
        self.num_epochs = num_epochs
        self.save_hyperparameters('num_epochs', 'num_classes')

        self.model = model.Model()

        self.criterion_list = nn.ModuleDict(
            dict(seg_loss=model.SegmentationLoss(), cluster_loss=model.ClusteringLoss()))

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

        loss = 0.
        for loss_name, criterion in self.criterion_list.items():
            loss_curr = criterion(target, **output)
            loss = loss_curr + loss
            self.log(f'loss/train_{loss_name}', loss_curr)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        with torch.no_grad():
            output = self(data)

            loss = 0.
            for loss_name, criterion in self.criterion_list.items():
                loss_curr = criterion(target, **output)
                loss = loss_curr + loss
                self.log(f'loss/val_{loss_name}', loss_curr)

            pred = predictSegment(**output)
            acc = (target == pred).sum() / float(torch.numel(target))
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
pl_model = ErfnetModule(1, 100)
trainer = pl.Trainer(gpus=1)
# trainer = pl.Trainer(fast_dev_run=True, gpus=1)

# %%
trainer.fit(pl_model, data_module)

# %%
loader = iter(data_module.val_dataloader())

# %%
img, labels = next(loader)
with torch.no_grad():
    output = pl_model(img)
    pred = model.predictSegment(**output)

# # plt.matshow(labels[0])
# # plt.colorbar()
# # plt.imshow(img[0].sum(axis=0))

# %%
frame_id = 5
fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(15,15))
ax0.imshow(img[frame_id].sum(axis=0))
ax1.matshow(labels[frame_id])
# ax2.matshow(torch.sigmoid(output['lane_segment'][frame_id]))
ax2.matshow(pred[frame_id])
# plt.savefig(f'seg_{frame_id}.png')

# %%
embedding = output['embedding_vector'][frame_id]
img_list = model.visualizeEmbedding(labels[frame_id], embedding)

fig, ax = plt.subplots(len(img_list) + 1, figsize=(15,15))
for i, v in enumerate(img_list):
    ax[i].imshow(v)
ax[-1].imshow(labels[frame_id])
# plt.savefig(f'embedding_{frame_id}.png')



# %%
