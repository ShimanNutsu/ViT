from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from src.modules.img_patches import ImgPatches
from src.modules.transformer import Transformer


def posemb_sincos_2d(patches, temperature=10000):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class ViT(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("ViT")

        # model options
        parser.add_argument("--lr", type=float, default=0.00025)
        parser.add_argument("--img_size", type=int, default=64)

        return parent_parser

    def __init__(self,
                 img_size=64,
                 patch_size=16,
                 in_ch=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 drop_rate=0.3,
                 lr: float = 0.00025,
                 **kwargs):
        super().__init__()
        self.lr = lr
        self.embed_dim = embed_dim
        self.img_patches = ImgPatches(in_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size)
        self.learnable_class_embeddings = nn.Parameter(torch.ones((1, 1, embed_dim)))
        self.transformer = Transformer(depth, embed_dim, num_heads, mlp_ratio, drop_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.save_hyperparameters()

    def forward(self, x):
        # Split into patches
        x = self.img_patches(x)

        # Concat with learnable class embeddings
        expanded_class_embeddings = self.learnable_class_embeddings.data.expand(x.shape[0], 1, x.shape[-1])
        x = torch.concat((x, expanded_class_embeddings), dim=1)

        # Perform position encoding
        x = x + posemb_sincos_2d(x)
        return self.classifier(self.transformer(x)[:, 0])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                        epochs=self.hparams['max_epochs'],
                                                        steps_per_epoch=self.hparams['steps_per_epoch'],
                                                        pct_start=0.2)
        return {"optimizer": optimizer,
                "lr_scheduler": {'scheduler': scheduler,
                                 'interval': 'step',
                                 'frequency': 1}, }

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.train_acc(y_pred, y)
        self.log('Train acc', self.train_acc, on_step=True)
        self.log('Train CE', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.val_acc(y_pred, y)
        self.log('Train acc', self.val_acc, on_step=True)
        self.log('Validation CE', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('Test CE', loss)

    def loss_fn(self, y_pred, y):
        return nn.functional.cross_entropy(y_pred, y)


if __name__ == '__main__':
    vit = ViT()
    x = torch.ones(1, 3, 224, 224)
    vit(x)
