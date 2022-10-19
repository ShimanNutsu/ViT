import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.model.img_patches import ImgPatches


class ViT(pl.LightningModule):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_ch=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 drop_rate=0.3):
        super().__init__()
        self.embed_dim = embed_dim

        self.img_patches = ImgPatches(in_ch=in_ch, embed_dim=self.embed_dim, patch_size=patch_size)
        self.learnable_class_embeddings = nn.Parameter(torch.ones((1, 1, embed_dim)))
        self.pos = nn.Parameter(torch.ones((1, 197, embed_dim)))

        pass

    def forward(self, x):
        # Split into patches
        x = self.img_patches(x)

        # Concat with learnable class embeddings
        expanded_class_embeddings = self.learnable_class_embeddings.data.expand(x.shape[0], 1, x.shape[-1])
        x = torch.concat((x, expanded_class_embeddings), dim=1)

        # Perform position encoding
        x = x + self.pos.data

        return x

    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
