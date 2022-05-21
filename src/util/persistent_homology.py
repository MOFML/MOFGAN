from typing import List

import numpy
import cripser
from numpy import float64
from torch import Tensor


def compute(grid: Tensor):
    assert grid.shape == (32, 32, 32)

    births = []
    deaths = []
    lifetimes = []

    result: numpy.ndarray = cripser.computePH(grid, maxdim=2, location="lifetime")

    for row in result:
        dim, birth, death = row[0], row[1], row[2]
        assert int(dim) == dim
        dim: int = int(dim)
        birth: float64
        death: float64

        if dim == 2:
            # print(birth, death, type(birth), type(death))
            births.append(birth)
            deaths.append(death)
            lifetimes.append(death - birth)

    return births, deaths, lifetimes


def compute_all(grids: List[Tensor]):
    global_births = []
    global_deaths = []
    global_lifetimes = []

    for grid in grids:
        births, deaths, lifetimes = compute(grid)
        global_births += births
        global_deaths += deaths
        global_lifetimes += lifetimes

    return global_births, global_deaths, global_lifetimes
