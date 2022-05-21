import copy
import hashlib
import inspect
import pickle
import time
from typing import Tuple, Callable, List

import numpy as np
import ray.util.multiprocessing
import torch
import wandb
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.mof_dataset import MOFDataset
from gan import training_config
from gan.training_config import TrainingConfig, DatasetType
from mofs import mof_stats, mof_properties
from util import cache

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GANMonitor:

    def __init__(self, train_config: TrainingConfig, image_shape: Tuple[int, int, int, int], data_loader: DataLoader,
                 latent_vector_generator: Callable[[int, bool], Tensor],
                 generator: Module, critic: Module,
                 generator_optimizer: Optimizer, critic_optimizer: Optimizer,
                 dataset_transformer: Callable[[Tensor], Tensor], enable_wandb: bool):
        self.train_config = train_config
        self.image_shape = image_shape
        self.batches_per_epoch = len(data_loader)
        self.latent_vector_generator = latent_vector_generator
        self.generator = generator
        self.critic = critic
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
        self.dataset_transformer = dataset_transformer
        self.enable_wandb = enable_wandb
        self.real_hcs: List[float] = []

        # batch_iterator = iter(data_loader)
        # for i in range(2):  # Batches to sample
        #     batch: Tensor = next(batch_iterator)
        #     sample_batch_path = (training_config.root_folder / f'real_sample_{i}.p')
        #     with sample_batch_path.open('wb+') as f:
        #         print(f"Saving real sample batch: {sample_batch_path}")
        #         pickle.dump(batch.cpu(), f)

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
        else:
            real_only_emd = None
        self.previous_real_pred = real_pred

        garbage_image: Tensor = torch.from_numpy(np.random.normal(0, 1, (batch.shape[0], np.prod(self.image_shape)))) \
            .float().requires_grad_(True).to(device).view(batch.shape[0], *self.image_shape)
        garbage_pred = self.critic(garbage_image)

        # NOTE: Garbage EMD should theoretically be very high relative to generated/real,
        # but we're not training to maximize that, only between generated, so I guess it makes sense?
        real_garbage_emd = abs(garbage_pred.mean() - real_pred.mean()).item()

        if self.enable_wandb and self.train_both_count % 5 == 0:
            wandb.log({"Negative Critic Loss": -d_loss, "Generator Loss": g_loss,
                       "Real/Generated EMD":   real_generated_emd,
                       "Real/Random EMD":      real_garbage_emd,
                       "Real/Real EMD":        real_only_emd})

        print(f"[Epoch {self.epoch}/{self.train_config.epochs}]".ljust(16)
              + f"[Batch {self.epoch_batch_index}/{self.batches_per_epoch}] ".ljust(14)
              + f"[-C Loss: {'{:.4f}'.format(-d_loss).rjust(11)}] "
              + f"[G Loss: {'{:.4f}'.format(g_loss).rjust(11)}] "
              + f"[Wasserstein Distance: {round(real_generated_emd, 3)}] "
              + f"[Garbage/Real EMD: {round(real_garbage_emd, 3)}] "
              + f"[Real Only EMD: {round(real_only_emd, 3) if real_only_emd else 'N/A'}]".ljust(14))
        self.train_both_count += 1

    def on_iteration_complete(self, generated_images: Tensor):
        save_id = str(self.global_batch_index).zfill(5)

        if self.global_batch_index % self.train_config.sample_interval == 0:
            save_start_time = time.time()
            save_path = training_config.images_folder / f"{save_id}.p"
            with open(save_path, "wb+") as f:
                pickle.dump(generated_images.cpu(), f)
            print(f"SAVED {save_path}  ({round(time.time() - save_start_time, 3)}s)")

            if self.global_batch_index % (10 * self.train_config.sample_interval) == 0:
                save_start_time = time.time()
                (training_config.states_folder / save_id).mkdir(exist_ok=True, parents=True)
                generator_state = {k: v.cpu() for k, v in self.generator.state_dict().items()}
                critic_state = {k: v.cpu() for k, v in self.critic.state_dict().items()}

                torch.save(generator_state, training_config.states_folder / save_id / 'generator.p')
                torch.save(self.generator_optimizer.state_dict(), training_config.states_folder / save_id / 'generator_optimizer.p')
                torch.save(critic_state, training_config.states_folder / save_id / 'critic.p')
                torch.save(self.critic_optimizer.state_dict(), training_config.states_folder / save_id / 'critic_optimizer.p')
                print(f"SAVED MODEL STATES ({round(time.time() - save_start_time, 3)}s)")

        # if self.enable_wandb and self.epoch_batch_index % (self.train_config.sample_interval * 2) == 0:
        # if self.epoch_batch_index % (self.train_config.sample_interval * 2) == 0:
        #     print("Checking HC distribution...")
        #
        #     # self.real_hcs = [1, 2, 3]
        #     transform_code: str = inspect.getsource(self.dataset_transformer)
        #     real_hc_cache_key = ['real_hcs', transform_code]
        #     self.real_hcs = cache.get(real_hc_cache_key)
        #
        #     if not self.real_hcs:
        #         print("Loading full real MOF dataset...")
        #         full_dataset = MOFDataset.load(training_config.datasets[DatasetType.FULL])
        #         print("Transforming full dataset")
        #         full_dataset.transform_(self.dataset_transformer)  # Need to compare to real HCs under same energy grid transformation
        #
        #         print("Computing real henry constants...")
        #         start = time.time()
        #         self.real_hcs = [mof_properties.get_henry_constant(mof[0]) for mof in full_dataset.mofs]
        #         print(f"DONE: {round(time.time() - start, 2)}s")
        #
        #         print("Saving...")
        #         cache.store(real_hc_cache_key, self.real_hcs)
        #         transform_code_hash: str = hashlib.sha1(transform_code.encode('utf-8')).hexdigest()
        #         with (training_config.root_folder / f"real_transformed_hcs-{transform_code_hash}.txt").open('w+') as f:
        #             f.write("\n".join(str(x) for x in self.real_hcs))  # Save real HC distribution
        #
        #     generator_clone = self.generator.__class__()
        #     generator_clone.load_state_dict(copy.deepcopy(self.generator.state_dict()))
        #     # generator_clone.cpu()
        #     with ray.util.multiprocessing.Pool(35) as pool:
        #         def generate_sample_hcs(latent_vector: Tensor):
        #             sample_hcs = []
        #             for hc_sample_mof in generator_clone(latent_vector):
        #                 sample_hcs.append(mof_properties.get_henry_constant(hc_sample_mof[0]))
        #             return sample_hcs
        #
        #         print("Computing current generated HC distribution")
        #         hc_check_start = time.time()
        #         request = [self.latent_vector_generator(110, True) for _ in range(106)]  # Same size as real = 11660
        #         hc_samples = list(tqdm(pool.imap(generate_sample_hcs, request),
        #                                total=len(request), ncols=80, unit='batches'))
        #         hc_samples = [x for sample in hc_samples for x in sample]
        #         print(f"DONE: {round(time.time() - hc_check_start, 2)}s")
        #     # for j in range(1166):
        #     # for hc_sample_mof in generator_clone(self.latent_vector_generator(10)):  # .cpu()
        #     #     hcs.append(mof_properties.get_henry_constant(hc_sample_mof[0]))
        #
        #     hc_result_path = training_config.metrics_folder / save_id / "henry_constant.txt"
        #     hc_result_path.parent.mkdir(parents=True, exist_ok=True)
        #     with hc_result_path.open('w+') as f:
        #         f.write("\n".join(str(x) for x in hc_samples))
        #
        #     hc_emd = mof_stats.scale_invariant_emd(hc_samples, self.real_hcs)
        #
        #     if self.enable_wandb:
        #         wandb.log({"HC Distribution EMD": hc_emd})
        #     else:
        #         print(f"HC Distribution EMD: {hc_emd}")
        #     print(f"HC Check Time: {round(time.time() - hc_check_start, 1)}s")


def filter_source_code(source_code: str):
    # Remove comments and print statements
    lines = [line for line in source_code.splitlines()
             if not line.strip().startswith('#') and not line.strip().startswith('print(')]

    # Remove for loops without a body
    lines = [lines[i] for i in range(len(lines)) if not (
            lines[i].strip().startswith('for') and i < len(lines) - 1 and lines[i + 1].strip() == "")]

    # Recombine lines
    filtered_source_code = "\n".join(line for line in lines).strip()

    # Max of 2 line separators
    while '\n\n\n' in filtered_source_code:
        filtered_source_code = filtered_source_code.replace('\n\n\n', '\n\n')

    return filtered_source_code
