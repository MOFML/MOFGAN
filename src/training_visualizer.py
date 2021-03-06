import json
import pickle
from enum import Enum

from torch import Tensor

import project_config
from gan import training_config
from util import utils, experimental


# REAL_SPHERE = 3
# HENRY_CONSTANT = 5
# HENRY_CONSTANT_TIMELINE = 6
# ROTATED_MOF = 9
# SCALED_MOF = 10
# BLURRED_MOF = 11
# ENERGY_DISTRIBUTION = 12

class Mode(Enum):
    REAL_MOF = 1
    REAL_MOF_TRANSFORMED = 2
    GENERATED_MOF = 3


# mode = Mode.REAL_MOF
# mode = Mode.REAL_MOF_TRANSFORMED
mode = Mode.GENERATED_MOF

image = '21000.p'
training_instance = training_config.instance_name
save_path = project_config.local.sample_save_path


def main():
    if mode == Mode.REAL_MOF:
        download_samples(f'real_mof_sample.p')
    elif mode == Mode.REAL_MOF_TRANSFORMED:
        download_samples(f'real_mof_sample.p', transform=True)
    elif mode == Mode.GENERATED_MOF:
        download_samples(f'_training/EnergyPosition-{training_instance}/images/{image}')
    # elif mode == Mode.REAL_MOF_LOCAL:
    # elif mode == Mode.REAL_MOF_REMOTE_SAMPLE:


def download_samples(remote_path: str, transform=False):
    with utils.sftp_connection() as sftp:
        print(f"Downloading samples from: {remote_path}")
        byte_data: bytes = sftp.download_bytes(remote_path)
        data: Tensor = pickle.loads(byte_data)

        if transform:
            data = experimental.experimental_transform_function(data)
        print(f"Sample shape: {data.shape}")

        with open(save_path, 'w+') as f:
            json.dump(data.tolist()[:32], f, indent='\t')
        print(f"SAVED TO {save_path}")


if __name__ == '__main__':
    main()
