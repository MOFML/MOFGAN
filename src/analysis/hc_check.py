import numpy as np
import torch

import project_config
from dataset.mof_dataset import MOFDataset
from gan import mofgan
from mofs import mof_properties
from project_config import RESOURCE_PATH

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # with utils.sftp_connection() as c:
    #     generator_bytes = c.download_bytes(training_config.states_folder / '60000' / 'generator.p')
    #     pickle.load(g)

    generator = mofgan.Generator().to(device)
    # generator.to(device)

    # generator.load_state_dict(torch.load(training_config.states_folder / '60000' / 'generator.p'))
    # with open('generator-60k.pt', 'wb+') as f:
    #     torch.save(generator.cpu().state_dict(), f)
    # ------

    local_data_folder = RESOURCE_PATH.parent / 'data'
    generator.load_state_dict(torch.load(local_data_folder / "generator-60k.pt"))

    latent_dim = 1024

    hcs = []
    for i in range(1166):
        batch_size = 10
        random_vector = torch.from_numpy(np.random.normal(0, 1, (batch_size, latent_dim))).float()
        generated_mof = generator(random_vector)
        hc = mof_properties.get_henry_constant(generated_mof)
        hcs.append(hc)
        if i % 10 == 0:
            print(i, hc)

    dataset = MOFDataset.load(f"{project_config.local.root}/mof_dataset.pt")
    print("Loaded dataset!")

    bins = 80
    from matplotlib import pyplot as plt

    dataset.transform_(mofgan.data_transform_function)
    hcs = [mof_properties.get_henry_constant(mof[0]) for mof in dataset.mofs]
    print("BOUNDS:", min(hcs), max(hcs))
    plt.title(f"Real Henry Constant Distribution: ({bins} bins)")

    plt.hist(hcs, bins=bins, color='green', alpha=0.8)
    plt.show()


if __name__ == '__main__':
    main()
