import os
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Dict

model_name = "EnergyPosition"
instance_name = "2Channel"

project_root_folder = Path()


def get_training_folder():
    return project_root_folder / '_training' / f"{model_name}-{instance_name}"  # TODO: No more timestamp here. Need to make sure we log everything important with wandb


root_folder = get_training_folder()
images_folder = root_folder / "images"
states_folder = root_folder / "states"
metrics_folder = root_folder / "metrics"


class TrainingConfig(NamedTuple):
    epochs: int = 200
    latent_dim: int = 2048  # Dimensionality of the latent space
    generator_train_interval: int = 5  # number of training steps for discriminator per iteration
    sample_interval: int = 100
    batch_size: int = 16

    critic_learning_rate: float = 0.0004
    generator_learning_rate: float = 0.0004
    # critic_learning_rate: float = 0.0001
    # generator_learning_rate: float = 0.0001
    lambda_gp: int = 10  # Loss weight for gradient penalty
    adam_b1: float = 0.5
    adam_b2: float = 0.9  # or 0.999
    # clip_value: float = 0.01


def create_directories():
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(states_folder, exist_ok=True)


class DatasetType(Enum):
    FULL = 1
    TRAIN = 2
    TEST = 3


datasets: Dict[DatasetType, str] = {
    DatasetType.FULL:  '_datasets/mof_dataset_2c.pt',
    DatasetType.TRAIN: '_datasets/mof_dataset_2c_train.pt',
    DatasetType.TEST:  '_datasets/mof_dataset_2c_test.pt',
}
