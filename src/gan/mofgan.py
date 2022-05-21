import inspect
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import project_config
import util
from dataset.mof_dataset import MOFDataset
from gan import training_config, gan_monitor
from gan.gan_logger import GANLogger
from gan.gan_monitor import GANMonitor
from gan.training_config import TrainingConfig, DatasetType
from testing.test_dataset import TestDataset
from util import transformations, utils

config = TrainingConfig()
GANLogger.log(config)

channels = 2
grid_size = 32
image_shape = (channels, grid_size, grid_size, grid_size)

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        kernel_size = 4
        stride = 2
        padding = 1

        self.g_latent_to_features = nn.Sequential(
            nn.Linear(config.latent_dim, 8 * grid_size * channels * 2 * 2 * 2),
            # nn.Sigmoid(),
            # nn.SiLU(),
        )

        self.g_features_to_image = nn.Sequential(
            nn.ConvTranspose3d(channels * grid_size * 8, grid_size * 4, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size * 4),
            # nn.ReLU(),
            nn.SiLU(),  # Swish

            nn.ConvTranspose3d(grid_size * 4, grid_size * 2, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size * 2),
            # nn.ReLU(),
            nn.SiLU(),

            nn.ConvTranspose3d(grid_size * 2, grid_size, kernel_size, stride, padding),
            nn.BatchNorm3d(grid_size),
            # nn.ReLU(),
            nn.SiLU(),

            nn.ConvTranspose3d(grid_size, channels, kernel_size, stride, padding),
            # nn.Tanh(),
            # nn.Sigmoid(),  # nn.SiLU(),
        )

    def forward(self, z):
        # print("ZIN:", z.shape) # [Batch x LatentDim]
        z = self.g_latent_to_features(z)
        # print("ZIN2:", z.shape)
        z = z.view(z.shape[0], channels * grid_size * 8, 2, 2, 2)
        # print("ZIN3:", z.shape)
        return self.g_features_to_image(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        kernel = (5, 5, 5)
        stride = (2, 2, 2)
        padding = 2

        # For different paddings: Final flattened size is [256, 2048, 16384, 55296, 131072, 256000, 442368, 702464] (K=5, S=2)
        padding_scales = {1: 1, 2: 8, 3: 64, 4: 216, 5: 512, 6: 1000, 7: 1728, 8: 2744}

        self.c_image_to_features = nn.Sequential(  # DON'T USE BATCH NORM WITH GP
            nn.Conv3d(channels, grid_size, kernel, stride, padding=padding, padding_mode='circular'),
            # nn.LayerNorm([grid_size, 16, 16, 16]),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.50),  # TODO: What's the best place for this?

            nn.Conv3d(grid_size, grid_size * 2, kernel, stride, padding=padding, padding_mode='circular'),
            # nn.LayerNorm([grid_size * 2, 8, 8, 8]),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 2, grid_size * 4, kernel, stride, padding=padding, padding_mode='circular'),
            # nn.LayerNorm([grid_size * 4, 4, 4, 4]),
            nn.LeakyReLU(0.2),

            nn.Conv3d(grid_size * 4, grid_size * 8, kernel, stride, padding=padding, padding_mode='circular'),
            # nn.LayerNorm([grid_size * 8, 2, 2, 2]),
            nn.LeakyReLU(0.2),
            # nn.Sigmoid(),

            # Flatten then linear
        )

        self.c_features_to_score = nn.Sequential(
            nn.Linear((grid_size * 8) * padding_scales[padding], 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):  # Input: [batch_size x channels x grid_size x grid_size x grid_size]
        # print("DISC INPUT:", x.shape)
        # import torch.nn.functional as functional
        # print("PADDING")
        # functional.pad(x, (2, 2, 2, 2, 2, 2), mode='circular')
        # print("DONE PADDING")
        x = self.c_image_to_features(x)
        # print("DISC OUTPUT:", x.shape)
        # print("RESHAPED:", x.view(x.shape[0], -1).shape)
        x = self.c_features_to_score(x.view(x.shape[0], -1))
        return x


def init_weights(m):
    classname = m.__class__.__name__
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def compute_gradient_penalty(disc, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = torch.from_numpy(np.random.random((real_samples.size(0), 1, 1, 1, 1))).float().to(device)  # TODO: [0,1)?
    # Get random interpolation between real and fake samples
    # interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = (alpha * real_samples + ((-alpha + 1) * fake_samples)).requires_grad_(True)
    d_interpolates = disc(interpolates)
    # fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = torch.ones(real_samples.shape[0], 1).to(device)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def get_bounds(tensor: Tensor):
    return torch.min(tensor).item(), torch.max(tensor).item()


def data_transform_function(mofs: Tensor) -> Tensor:
    print("Extracting channels...")
    energy_grids = mofs[:, 0].float()
    position_grids = mofs[:, 1].float()

    print("Original Data:")
    print(f"\tEnergy Bounds: {get_bounds(energy_grids)}")
    print(f"\tPosition Bounds: {get_bounds(position_grids)} ")

    # energy_grids = torch.from_numpy(np.interp(energy_grids, [-5200, 5200], [0, 1])).float()
    # position_grids = torch.from_numpy(np.interp(position_grids, [0, 7.5], [0, 1])).float()

    # energy_grids = transformations.scale_log(energy_grids)  # (-2.3, 3.7)
    # energy_grids = (energy_grids - 12.5) / 10  # Single log scale
    # # energy_grids = (energy_grids - 2) / 1.5  # Double log scale (Note: double scaling can cause out of memory issues)
    # position_grids = (position_grids * 20) - 0.15

    energy_grids = transformations.scale_log(energy_grids)
    # energy_grids = torch.from_numpy(np.interp(energy_grids, [-5200, 5200], [0, 1])).float()

    print("Post Numeric Transformation Bounds:")
    print(f"\tEnergy: {get_bounds(energy_grids)}")
    print(f"\tPositions: {get_bounds(position_grids)}")

    energy_grids = transformations.standardize(energy_grids)
    # position_grids = transformations.standardize(position_grids)

    # energy_grids = torch.from_numpy(np.interp(energy_grids, [-9, 42], [-1, 1])).float()
    # position_grids = torch.from_numpy(np.interp(position_grids, [0, 7.5], [-1, 1])).float()

    # energy_grids = torch.from_numpy(np.interp(energy_grids, [-2.3, 3.7], [-1, 1])).float()
    # position_grids = torch.from_numpy(np.interp(position_grids, [0, 7.5], [-1, 1])).float()
    # energy_grids = torch.from_numpy(np.interp(energy_grids, [-2.3, 3.7], [-1, 1])).float()
    # position_grids = torch.from_numpy(np.interp(position_grids, [0, 7.5], [-1, 1])).float()

    result = torch.stack([energy_grids, position_grids], dim=1)
    print("\n----- FINAL -----")
    for name, channel in {"Energy": energy_grids, "Positions": position_grids}.items():
        print(f"{name}:")
        print(f"\tBounds: {get_bounds(channel)}")
        print(f"\tMean: {channel.mean()}")
        print(f"\tSTD: {channel.std()}")

    print(f"Training data shape: {result.shape}")

    return result


def generate_random_latent_vector(batch_size: int, cpu=False) -> Tensor:
    if cpu:
        return torch.from_numpy(np.random.normal(0, 1, (batch_size, config.latent_dim))) \
            .float().cpu()
    else:
        return torch.from_numpy(np.random.normal(0, 1, (batch_size, config.latent_dim))) \
            .float().requires_grad_(True).to(device)


# ----------
#  Training
# ----------

def train():
    start = time.time()
    # TODO: Probably want to rotate after loading instead of storing rotations. Should use the same memory

    dataset_type = DatasetType.TRAIN
    # dataset_type = DatasetType.TEST
    dataset_path: str = training_config.datasets[dataset_type]
    enable_wandb: bool = (dataset_type == DatasetType.TRAIN)

    # data_loader = MOFDataset.get_data_loader("_datasets/mof_dataset_test_rotate.pt", batch_size=config.batch_size, shuffle=True)

    GANLogger.log(f"SOURCE DATASET: {dataset_path}")

    title = f"MOF WGAN GP - GLR: {config.generator_learning_rate}, DLR: {config.critic_learning_rate}, S={image_shape}, BS={config.batch_size}"
    GANLogger.init(title, training_config.root_folder)

    print("Loading dataset...")

    # dataset = MOFDataset.load(dataset_path)
    # dataset.transform_(data_transform_function)
    # if dataset_type == DatasetType.TRAIN:
    #     dataset = dataset.augment_rotations()

    dataset = TestDataset.load('test_sphere_cube_dataset.pt')
    print("LOADED:", dataset.data.shape)

    data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)

    print(f"LOAD TIME: {round((time.time() - start), 2)}s")
    transform_code: str = inspect.getsource(data_transform_function)
    GANLogger.log("Transform Function:", console=False)
    GANLogger.log(*["\t" + line for line in transform_code.splitlines()], console=False)

    # exit()

    generator = Generator().to(device)
    critic = Discriminator().to(device)
    generator.apply(init_weights)
    critic.apply(init_weights)
    GANLogger.log(generator, critic)

    training_config.create_directories()

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(),
                                           lr=config.generator_learning_rate, betas=(config.adam_b1, config.adam_b2))
    critic_optimizer = torch.optim.Adam(critic.parameters(),
                                        lr=config.critic_learning_rate, betas=(config.adam_b1, config.adam_b2))

    if enable_wandb:
        wandb_meta = dict(config._asdict())

        wandb.init(project=training_config.model_name, name=f"{training_config.instance_name}-{int(time.time() * 1000)}",
                   entity=project_config.wandb.user, config=wandb_meta)

        wandb.termlog(str(generator))
        wandb.termlog(str(critic))
        wandb.termlog(gan_monitor.filter_source_code(transform_code))

        wandb.watch(models=[generator, critic], log_freq=400)

    monitor = GANMonitor(train_config=config, image_shape=image_shape,
                         latent_vector_generator=generate_random_latent_vector,
                         data_loader=data_loader, generator=generator, critic=critic,
                         generator_optimizer=generator_optimizer, critic_optimizer=critic_optimizer,
                         dataset_transformer=data_transform_function, enable_wandb=enable_wandb)

    batch_index = 0
    for epoch in range(1, config.epochs + 1):
        for i, images in enumerate(data_loader):
            monitor.set_iteration(epoch=epoch, global_batch_index=batch_index, epoch_batch_index=i)

            real_images = images.to(device).requires_grad_(True)
            current_batch_size = images.shape[0]

            # ---------------------
            #  Train Discriminator
            # ---------------------

            critic_optimizer.zero_grad()

            # Sample noise as generator input
            noise = generate_random_latent_vector(current_batch_size)

            generated_images: Tensor = generator(noise)  # Generate a batch of images

            real_pred: Tensor = critic(real_images)
            generated_pred: Tensor = critic(generated_images)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_images.data, generated_images.data)

            # Adversarial loss: We want the critic to maximize the separation between fake and real
            d_loss: Tensor = generated_pred.mean() - real_pred.mean() + config.lambda_gp * gradient_penalty

            d_loss.backward()
            critic_optimizer.step()

            generator_optimizer.zero_grad()

            # Train the generator every N batches
            if i % config.generator_train_interval == 0:
                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                generated_images: Tensor = generator(noise)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                generated_pred: Tensor = critic(generated_images)
                g_loss: Tensor = -torch.mean(generated_pred)

                g_loss.backward()
                generator_optimizer.step()

                monitor.train_both(batch=images, real_pred=real_pred, generated_pred=generated_pred,
                                   g_loss=g_loss.item(), d_loss=d_loss.item())

            monitor.on_iteration_complete(generated_images)
            batch_index += 1


# Weight clipping:
# for p in discriminator.parameters():
#     p.data.clamp_(-clip_value, clip_value)

if __name__ == '__main__':
    # TODO:
    #  Try Tanh in generator output with E = [-1, 1] and P = [0, 1] (or P also = [-1, 1])
    #  Hyperparameter optimization. Latent: [16, 64, 256, 1024], Batch Size: [16, 64, 256], Epochs = 10.
    #  Adam B2, LR, Kernel Size, Padding, Dropout Rate, Weight Initialization

    train()
    # asyncio.run(main())
