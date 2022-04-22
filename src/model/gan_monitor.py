import pickle
import time
from typing import Tuple, Callable

import numpy as np
import torch
import wandb
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import config
from domain import mof_stats, mof_properties
from model import training_config
from model.training_config import Config

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GANMonitor:

    def __init__(self, train_config: Config, image_shape: Tuple[int, int, int, int], batches_per_epoch: int,
                 latent_vector_generator: Callable[[int], Tensor],
                 generator: Module, critic: Module,
                 generator_optimizer: Optimizer, critic_optimizer: Optimizer):
        self.train_config = train_config
        self.image_shape = image_shape
        self.batches_per_epoch = batches_per_epoch
        self.latent_vector_generator = latent_vector_generator
        self.generator = generator
        self.critic = critic
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer

        self.epoch = 0
        self.global_batch_index = 0
        self.epoch_batch_index = 0

        self.previous_real_pred = None
        self.train_both_count = 0

    def set_iteration(self, epoch: int, global_batch_index: int, epoch_batch_index: int):
        self.epoch = epoch
        self.global_batch_index = global_batch_index
        self.epoch_batch_index = epoch_batch_index

    def train_both(self, batch: Tensor, real_pred: Tensor, generated_pred: Tensor, g_loss: float, d_loss: float):
        # GANLogger.update(-d_loss.item(), g_loss.item())

        real_generated_emd = abs(generated_pred.mean() - real_pred.mean()).item()

        if self.previous_real_pred is not None:
            real_only_emd = abs(real_pred.mean() - self.previous_real_pred.mean()).item()
            print(f"REAL ONLY EMD: {real_only_emd}")
        else:
            real_only_emd = None
        self.previous_real_pred = real_pred

        garbage_image: Tensor = torch.from_numpy(np.random.normal(0, 1, (batch.shape[0], np.prod(self.image_shape)))) \
            .float().requires_grad_(True).to(device).view(batch.shape[0], *self.image_shape)
        garbage_pred = self.critic(garbage_image)

        # NOTE: Garbage EMD should theoretically be very high relative to generated/real,
        # but we're not training to maximize that, only between generated, so I guess it makes sense?
        real_garbage_emd = abs(garbage_pred.mean() - real_pred.mean()).item()
        print("GARBAGE/REAL EMD:", real_garbage_emd)

        if self.train_both_count % 5 == 0:
            wandb.log({"Negative Critic Loss": -d_loss, "Generator Loss": g_loss,
                       "Real/Generated EMD":   real_generated_emd,
                       "Real/Random EMD":      real_garbage_emd,
                       "Real/Real EMD":        real_only_emd})

        print(f"[Epoch {self.epoch}/{self.train_config.epochs}]".ljust(16)
              + f"[Batch {self.epoch_batch_index}/{self.batches_per_epoch}] ".ljust(14)
              + f"[-C Loss: {'{:.4f}'.format(-d_loss).rjust(11)}] "
              + f"[G Loss: {'{:.4f}'.format(g_loss).rjust(11)}] "
              + f"[Wasserstein Distance: {round(real_generated_emd, 3)}]")
        self.train_both_count += 1

    def on_iteration_complete(self, generated_images: Tensor):
        if self.global_batch_index % self.train_config.sample_interval == 0:
            save_start_time = time.time()
            save_id = str(self.global_batch_index).zfill(5)
            save_path = training_config.images_folder / f"{save_id}.p"
            with open(save_path, "wb+") as f:
                pickle.dump(generated_images.cpu(), f)
            print(f"SAVED {save_path}  ({round(time.time() - save_start_time, 3)}s)")

            if self.global_batch_index % (10 * self.train_config.sample_interval) == 0:
                save_start_time = time.time()
                (training_config.states_folder / save_id).mkdir(exist_ok=True, parents=True)
                torch.save(self.generator.state_dict(), training_config.states_folder / save_id / 'generator.p')
                torch.save(self.generator_optimizer.state_dict(), training_config.states_folder / save_id / 'generator_optimizer.p')
                torch.save(self.critic.state_dict(), training_config.states_folder / save_id / 'critic.p')
                torch.save(self.critic_optimizer.state_dict(), training_config.states_folder / save_id / 'critic_optimizer.p')
                print(f"SAVED MODEL STATES ({round(time.time() - save_start_time, 3)}s)")

        if self.epoch_batch_index % (self.train_config.sample_interval * 2) == 0:
            print("Checking HC distribution...")  # TODO: Parallelize this
            hc_check_start = time.time()
            hcs = []
            for j in range(1166):
                for hc_sample_mof in self.generator(self.latent_vector_generator(10)):
                    hcs.append(mof_properties.get_henry_constant(hc_sample_mof))
            print(f"Generated samples {round(time.time() - hc_check_start, 2)}s")
            with open(config.RESOURCE_PATH / 'real_henry_constant_scaled_mof.txt') as f:
                real_hcs = [float(x) for x in f.read().splitlines()]
            hc_emd = mof_stats.scale_invariant_emd(hcs, real_hcs)

            wandb.log({"HC Distribution EMD": hc_emd})
            print(f"HC Check Time: {time.time() - hc_check_start}s")
