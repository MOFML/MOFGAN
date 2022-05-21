import numpy as np
import torch
from torch import Tensor

from dataset.mof_dataset import MOFDataset
from util import transformations


def get_bounds(tensor: Tensor):
    return torch.min(tensor).item(), torch.max(tensor).item()


def experimental_transform_function(mofs: Tensor) -> Tensor:
    print("Extracting channels...")
    energy_grids = mofs[:, 0].float()
    position_grids = mofs[:, 1].float()

    print("Original Data:")
    print(f"\tEnergy Bounds: {get_bounds(energy_grids)}")
    print(f"\tPosition Bounds: {get_bounds(position_grids)}")

    energy_grids = transformations.scale_log(energy_grids)
    # position_grids = 8 * position_grids
    # position_grids = np.exp(position_grids) / 7

    # print("PG MEAN:", position_grids.mean())

    energy_grids = transformations.standardize(energy_grids)
    # position_grids = transformations.standardize(position_grids)

    print("Post Numeric Transformation Bounds:")
    print(f"\tEnergy: {get_bounds(energy_grids)}")
    print(f"\tPositions: {get_bounds(position_grids)}")

    # energy_grids = torch.from_numpy(np.interp(energy_grids, [-9, 42], [0, 1])).float()
    # position_grids = torch.from_numpy(np.interp(position_grids, [0, 7.5], [0, 1])).float()
    # position_grids = torch.from_numpy(np.interp(position_grids, [0, 42], [0, 1])).float()

    print("\n----- FINAL -----")
    for name, channel in {"Energy": energy_grids, "Positions": position_grids}.items():
        print(f"{name}:")
        print(f"\tBounds: {get_bounds(channel)}")
        print(f"\tMean: {channel.mean()}")
        print(f"\tSTD: {channel.std()}")

    result = torch.stack([energy_grids, position_grids], dim=1)

    print(f"Resulting shape: {result.shape}")
    return result


def test_transform():
    print("Loading dataset...")
    dataset = MOFDataset.load('_datasets/mof_dataset_2c.pt')
    # dataset = MOFDataset.load('_datasets/mof_dataset_2c_test.pt')

    dataset.transform_(experimental_transform_function)


if __name__ == '__main__':
    test_transform()
