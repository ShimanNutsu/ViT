from typing import Optional

import pytorch_lightning as pl
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms


class EuroSATDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def prepare_data(self):
        ds = EuroSAT(root=self.data_dir, download=True)
        print(len(ds))

    def setup(self, stage: Optional[str] = None):
        self.eurosat_test = EuroSAT(self.data_dir)
        self.eurosat_predict = EuroSAT(self.data_dir)
        eurosat_full = EuroSAT(self.data_dir, transform=self.transform)
        self.eurosat_train, self.eurosat_val = random_split(eurosat_full, [25000, 2000])

    def train_dataloader(self):
        return DataLoader(self.eurosat_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.eurosat_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.eurosat_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.eurosat_predict, batch_size=self.batch_size)


if __name__ == '__main__':
    module = EuroSATDataModule('./data/')
    module.prepare_data()
    module.setup()
    dl = module.train_dataloader()

    img = next(iter(dl))
    # print(img.shape)

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    import torchvision.transforms.functional as F

    plt.rcParams["savefig.bbox"] = 'tight'


    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach().numpy().transpose(1, 2 ,0)
            axs[0, i].imshow(img)
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()

    imgs = list(torch.unbind( img[0][:5], dim=0))
    show(imgs)