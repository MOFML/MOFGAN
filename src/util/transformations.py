import numpy as np
import torch
from torch import Tensor


def standardize(data: Tensor):
    return (data - torch.mean(data)) / torch.std(data)


def scale_log(k):
    return np.sign(k) * np.log(abs(k) + 1)


def inverse_scale_log(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)


def main():
    print(scale_log(1234), scale_log(-1234))
    print(scale_log(0.1234), scale_log(-0.1234))
    print(round(inverse_scale_log(scale_log(1234)), 6), round(inverse_scale_log(scale_log(-1234)), 6))
    print(round(inverse_scale_log(scale_log(0.1234)), 6), round(inverse_scale_log(scale_log(-0.1234)), 6))


if __name__ == '__main__':
    main()
