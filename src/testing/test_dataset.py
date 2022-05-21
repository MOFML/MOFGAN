from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, data: Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def load(path: str | Path) -> TestDataset:
        with open(path, 'rb') as f:
            return torch.load(f)

    @staticmethod
    def get_data_loader(path: str, batch_size: int, shuffle: bool):
        return torch.utils.data.DataLoader(
            TestDataset.load(path),
            batch_size=batch_size,
            shuffle=shuffle,
        )
