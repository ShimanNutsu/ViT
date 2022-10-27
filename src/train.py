import sys
from pathlib import Path

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

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_SEED = 42
# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--path_to_dataset", type=Path,
                            default=Path(__file__).absolute().parent / "dataset" / "data",
                            help="Path to the dataset directory")

# Experiment parameters
program_parser.add_argument("--batch_size", type=int, default=2)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

# Add model specific args
parser = ViT.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Random
# ------------------------------------------------------------

seed_everything(args.seed, workers=True)

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------

project_name = 'vit_eurosat'
wandb_logger = WandbLogger(project=project_name, name='hello', log_model=True)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

datamodule = EuroSATDataModule(args.path_to_dataset)
datamodule.prepare_data()
datamodule.setup()

# model
print(args.path_to_dataset)
dict_args = vars(args)
dict_args['steps_per_epoch'] = len(datamodule.train_dataloader())
vit = ViT(**dict_args)
# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

monitor = 'Validation MSE'

# checkpoints
save_top_k = 1
checkpoint_callback = ModelCheckpoint(save_top_k=1)

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

devices = [int(args.devices)]
# trainer
trainer = pl.Trainer(accelerator='gpu',
                     devices=devices,
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     # precision=16,
                     deterministic=False)

if not len(args.from_checkpoint):
    args.from_checkpoint = None

# Train
trainer.fit(vit, datamodule=datamodule)

# TODO: cosine ecodings
# TODO: linear encodings
# TODO: learning rate
# TODO: scheduler one cycle linear 10k warmup
# TODO: gradient clipping
# TODO: Attention reduce projection dimension