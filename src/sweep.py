import yaml
import wandb
from train import train

from pathlib import Path

sweep_configuration = yaml.safe_load(Path('sweep.yaml').read_text())

print(sweep_configuration)

sweep_id = wandb.sweep(sweep=sweep_configuration, project='test_sweep')

wandb.agent(sweep_id, function=train)