import random
import time
from typing import List, Callable

import torch
from tqdm import tqdm

grid_size = 32
cx = cy = cz = grid_size // 2

GeneratorFunction = Callable[[int, int, int, int], float]


def sphere(radius, i, j, k) -> float:
    radius_squared = radius * radius
    d2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2

    if d2 < radius_squared:
        return 1 - d2 / radius_squared
        # sphere.append(1)
    else:
        return 0


def cube(length: int, i, j, k) -> float:
    # d = max(abs(i - cx), abs(j - cy), abs(k - cz))
    # if (abs(i - cx) < length or abs(k - cz) < length) and abs(j - cy) < length:

    ds = abs(i - cx), abs(j - cy), abs(k - cz)
    if max(ds) < length:
        d = min(ds)
        return 1 - d / length
    else:
        return 0


def generate_shape(size: int, generator: GeneratorFunction) -> List[float]:
    result = []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                result.append(generator(size, i, j, k))
    return result


def generate(samples: int, channels: List[GeneratorFunction]):
    print(torch.as_tensor([1, 2, 3, 4, 5, 6]))

    images = []

    start = time.time()
    for s in tqdm(range(samples), ncols=80, unit='grids'):
        # image = [0] * (grid_size ** 3)
        image = []
        for channel_generator in channels:
            size = random.randint(1, grid_size // 2)
            generated = generate_shape(size, channel_generator)
            # channel(radius)
            # print(len(generated))
            image += generated
            # for i in range(len(image)):
            #     image[i] = max(image[i], generated[i])
        images += image

    print(f"Created sphere dataset: {round((time.time() - start), 2)}s")
    c = torch.as_tensor(images).view(samples, len(channels), grid_size, grid_size, grid_size).float()
    print(c.shape)
    return torch.as_tensor(images).view(samples, len(channels), grid_size, grid_size, grid_size).float()
    # print(json.dumps(full, indent='\t'))


def generate_spheres2(samples: int, atoms: int):
    print(torch.as_tensor([1, 2, 3, 4, 5, 6]))

    flat = []
    data = []

    start = time.time()
    for s in range(samples):
        full = []

        rss = []
        cxs = []
        cys = []
        czs = []
        for atom in range(atoms):
            radius = random.randint(1, grid_size // 2)
            radius_squared = radius ** 2
            rss.append(radius_squared)
            cxs.append(random.randint(0, grid_size))
            cys.append(random.randint(0, grid_size))
            czs.append(random.randint(0, grid_size))

        for i in range(grid_size):
            js = []
            for j in range(grid_size):
                ks = []
                for k in range(grid_size):
                    for n in range(len(rss)):
                        if (i - cxs[n]) ** 2 + (j - cys[n]) ** 2 + (k - czs[n]) ** 2 < rss[n]:
                            flat.append(1)
                            ks.append(1)
                            break
                    else:
                        ks.append(0)
                        flat.append(0)
                js.append(ks)
            full.append(js)
        data.append(full)

    print(f"Created sphere dataset: {round((time.time() - start), 2)}s")
    c = torch.as_tensor(flat).view(samples, 1, grid_size, grid_size, grid_size).float()
    print(c.shape)
    return torch.as_tensor(flat).view(samples, 1, grid_size, grid_size, grid_size).float()
    # print(json.dumps(full, indent='\t'))


def create_dataset(samples: int):
    data = generate(samples, channels=[sphere, cube])
    with open(f'test_sphere_cube_dataset_{samples}.pt', 'wb+') as f:
        torch.save(data, f)


def main():
    # random.expovariate()
    create_dataset(7000)

    # result = generate(16, channels=[sphere, cube])
    # import training_visualizer
    # training_visualizer.save(result)
    print("DONE!")


if __name__ == '__main__':
    main()
