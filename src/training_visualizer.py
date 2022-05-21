import json
import pickle
from enum import Enum
from typing import List, Union

import torch
from torch import Tensor

import project_config
from gan import training_config
from util import utils, experimental, grid_scaling


# REAL_SPHERE = 3
# HENRY_CONSTANT = 5
# HENRY_CONSTANT_TIMELINE = 6
# ROTATED_MOF = 9
# BLURRED_MOF = 11
# ENERGY_DISTRIBUTION = 12

class Mode(Enum):
    REAL_MOF = 1
    REAL_MOF_TRANSFORMED = 2
    GENERATED_MOF = 3
    SCALED = 4


# mode = Mode.REAL_MOF
# mode = Mode.REAL_MOF_TRANSFORMED
mode = Mode.GENERATED_MOF
# mode = Mode.SCALED

image = '80000.p'
training_instance = training_config.instance_name
save_path = project_config.local.sample_save_path


def main():
    if mode == Mode.REAL_MOF:
        save(download_samples(f'real_mof_sample.p'))
    elif mode == Mode.REAL_MOF_TRANSFORMED:
        save(download_samples(f'real_mof_sample.p', transform=True))
    elif mode == Mode.GENERATED_MOF:
        save(download_samples(f'_training/EnergyPosition-{training_instance}/images/{image}'))
    elif mode == Mode.SCALED:
        samples = download_samples(f'real_mof_sample.p')
        sample = samples[0]
        scales = [32, 40, 50, 60, 70, 160]
        # scales = [32, 30, 25, 20, 15, 10]
        result = [torch.stack([grid_scaling.resize_3d(channel, scale) for channel in sample]).tolist()
                  for scale in scales]
        print(len(result), len(result[0]))

        # result = torch.stack([torch.stack([grid_scaling.resize_3d(channel, 64) for channel in sample])
        #                       for sample in samples])
        # print(f"Final size: {result.shape}")
        save(result)
        # elif mode == Mode.REAL_MOF_LOCAL:
    # elif mode == Mode.REAL_MOF_REMOTE_SAMPLE:


def download_samples(remote_path: str, transform=False) -> Tensor:
    with utils.sftp_connection() as sftp:
        print(f"Downloading samples from: {remote_path}")
        byte_data: bytes = sftp.download_bytes(remote_path)
        data: Tensor = pickle.loads(byte_data)

        if transform:
            data = experimental.experimental_transform_function(data)
        print(f"Sample shape: {data.shape}")
        return data


def save(data: Union[Tensor, List]):
    with open(save_path, 'w+') as f:
        list_data = data.tolist() if isinstance(data, Tensor) else data
        json.dump(list_data[:32], f, indent='\t')
    print(f"SAVED TO {save_path}")


if __name__ == '__main__':
    main()
