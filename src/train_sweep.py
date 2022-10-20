import sys
from pathlib import Path
import yaml

import wandb

sys.path.append("..")

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from argparse import ArgumentParser

from dataset.eurosat import EuroSATDataModule
from src.model.vit import ViT



def train():
    wandb.init()
    dict_args = dict(wandb.config)
    config = wandb.config


    seed_everything(dict_args['seed'], workers=True)

    datamodule = EuroSATDataModule(config.path_to_dataset)
    datamodule.prepare_data()
    datamodule.setup()
    dict_args['steps_per_epoch'] = len(datamodule.train_dataloader())
    vit = ViT(**dict_args)

    wandb_logger = WandbLogger()
    monitor = 'Val acc'

    # checkpoints
    save_top_k = 1
    checkpoint_callback = ModelCheckpoint(save_top_k=save_top_k, monitor=monitor,
                                          filename='best',
                                          auto_insert_metric_name=False,
                                          verbose=True)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # logger_callback= SlotAttentionLogger(val_samples=next(iter(val_dataset)))

    callbacks = [
        checkpoint_callback,
        # logger_callback,
        # swa,
        # early_stop_callback,
        lr_monitor,
    ]

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
    # trainer parameters
    profiler = 'simple'  # 'simple'/'advanced'/None

    devices = [int(config.devices)]
    # trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=devices,
                         max_epochs=config.max_epochs,
                         profiler=profiler,
                         callbacks=callbacks,
                         logger=wandb_logger,
                         deterministic=False)

    trainer.fit(vit, datamodule=datamodule)




# TODO: cosine ecodings
# TODO: linear encodings
# TODO: learning rate
# TODO: scheduler one cycle linear 10k warmup
# TODO: gradient clipping
# TODO: Attention reduce projection dimension
