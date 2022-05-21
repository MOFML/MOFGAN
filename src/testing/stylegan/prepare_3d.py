import pickle
import sys
from typing import Tuple

import lmdb
from torch import Tensor
from tqdm import tqdm

from dataset.mof_dataset import MOFDataset
from util import grid_scaling


def resize_worker(data: Tensor, sizes: Tuple[int]):
    return [grid_scaling.resize_3d(data, scale) for scale in sizes]


def prepare(db):
    sizes = [4, 8, 16, 32]

    d = MOFDataset.load('/home/kphill11/MOFGAN/_datasets/mof_dataset_2c_train.pt')
    files = [(index, mof) for index, mof in enumerate(d.mofs)]

    total = 0
    for i, images in enumerate(tqdm(files, ncols=80, unit='images', total=len(files))):
        for size, image in zip(sizes, images):
            key = f"{size}-{str(i).zfill(4)}".encode("utf-8")

            with db.begin(write=True) as txn:
                txn.put(key, pickle.dumps(image))
        total += 1

    with db.begin(write=True) as txn:
        txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: prepare_3d.py <output_folder>")
        exit()
    output_folder = args[0]
    # resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}

    with lmdb.open(output_folder, map_size=1024 ** 4, readahead=False) as db:
        prepare(db)


if __name__ == "__main__":
    main()
