import torch
import torch.nn as nn


class ImgPatches(nn.Module):
    def __init__(self,
                 in_ch: int = 3,
                 embed_dim: int = 768,
                 patch_size: int = 16):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, img):
        patches = self.patch_embed(img)
        patches = torch.flatten(patches, 2)
        patches = torch.transpose(patches, 2, 1)
        return patches


if __name__ == '__main__':
    img_patches = ImgPatches()
    x = torch.randn(10, 3, 224, 224)

    ans = img_patches(x)
    print(ans.shape)