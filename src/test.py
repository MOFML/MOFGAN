import numpy
import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch import Tensor

import project_config
from dataset.mof_dataset import MOFDataset
from util import transformations, grid_scaling, persistent_homology


def main():
    print("HI")
    t = torch.Tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24], [25, 26, 27]],
        [[28, 29, 30], [31, 32, 33], [34, 35, 56]],
    ])
    print(t[0][0][0])
    t[:, 0][0] = 999
    print(t[0][0][0])
    print(t[:, 0])

    # print(t.shape)
    # print(type(t.shape))
    # print(t.shape == (2, 3))
    # print(t.shape == Size([2, 3]))
    # print(Size([1, 2, 3]))


def test():
    def get_bounds(tensor: Tensor):
        return torch.min(tensor).item(), torch.max(tensor).item()

    print("Loading dataset...")
    # dataset = MOFDataset.load('_datasets/mof_dataset_2c.pt')
    dataset = MOFDataset.load('_datasets/mof_dataset_2c_test.pt')
    # for mof_name, mof in zip(dataset.mof_names, dataset.mofs):
    #     min_val = mof[1].min()
    #     if min_val < 0:
    #         print(mof_name, mof[1].min())

    grid_size = 32
    mofs = torch.stack([grid_scaling.resize_3d(mof[0], grid_size).unsqueeze(0) for mof in dataset.mofs])
    # mofs = torch.stack([mof[0].unsqueeze(0) for mof in dataset.mofs])

    print("Calculating...")

    # ssim_val = ssim(x, x, data_range=255, size_average=False)  # return (N,)
    # ms_ssim_val = ms_ssim(x, x, data_range=255, size_average=False)  # (N,)
    # min_val, max_val = get_bounds(mofs)
    # print(f"MIN: {min_val}")
    # print(f"MAX: {max_val}")

    rand_mof = torch.rand(200, 1, grid_size, grid_size, grid_size).float()
    mofs = torch.from_numpy(np.interp(transformations.scale_log(mofs), [-9, 42], [0, 1])).float()
    x = mofs[:200]
    y = mofs[200:400]

    print(x.shape)
    print(y.shape)
    print(rand_mof.shape)

    print(ms_ssim(x, y, data_range=1, win_size=1))  # 0.9937
    print(ssim(x, y, data_range=1))  # 0.0293

    print("FAKE:")
    print(ms_ssim(x, rand_mof, data_range=1, win_size=1))  # 0.9938
    print(ssim(x, rand_mof, data_range=1))  # 0.0081

    # img2 = torch.rand(400, 2, 32, 32, 32)
    # print(img1.shape)
    #
    # print(ms_ssim(img1, img2, win_size=1))
    # print(ssim(img1, img2))
    # dataset.transform_(experimental_transform_function)


def test_ph():
    dataset = MOFDataset.load('_datasets/mof_dataset_2c_test.pt')
    # dataset = MOFDataset.load(f"{project_config.local.root}/mof_dataset_2c_test.pt")

    global_births = []
    global_deaths = []
    global_lifetimes = []

    for mof_index in range(10):
        births, deaths, lifetimes = persistent_homology.compute(dataset.mofs[mof_index][0])
        global_births += births
        global_deaths += deaths
        global_lifetimes += lifetimes

    mof_index = 0
    births, deaths, lifetimes = persistent_homology.compute(dataset.mofs[mof_index][0])

    exit()
    import matplotlib.pyplot as plt
    plt.xlim(-5, 50)
    plt.ylim(-5, 50)
    plt.title(f'Persistent Homology of {dataset.mof_names[mof_index]}')
    plt.barh(range(len(lifetimes)), left=births, width=lifetimes)
    plt.yticks([])
    plt.xlabel('Lifetime')
    plt.show()


if __name__ == '__main__':
    # main()
    # test()
    test_ph()
