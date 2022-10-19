from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.modules.img_patches import ImgPatches
from src.modules.transformer import Transformer


class ViT(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("ViT")

        # model options
        parser.add_argument("--lr", type=float, default=0.00025)

        return parent_parser

    def __init__(self,
                 img_size=224,
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
        self.pos = nn.Parameter(torch.ones((1, 197, embed_dim)))
        self.transformer = Transformer(depth, embed_dim, num_heads, mlp_ratio, drop_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        # Split into patches
        x = self.img_patches(x)

        # Concat with learnable class embeddings
        expanded_class_embeddings = self.learnable_class_embeddings.data.expand(x.shape[0], 1, x.shape[-1])
        x = torch.concat((x, expanded_class_embeddings), dim=1)

        # Perform position encoding
        x = x + self.pos.data
        return self.classifier(self.transformer(x)[0])

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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(batch)
        loss = self.loss_fn(y_pred, y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(batch)
        loss = self.loss_fn(y_pred, y)

    def loss_fn(self, y_pred, y):
        return nn.functional.cross_entropy(y_pred, y)


if __name__ == '__main__':
    vit = ViT()
    x = torch.ones(1, 3, 224, 224)
    vit(x)
