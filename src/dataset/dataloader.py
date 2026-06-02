from typing import List, Dict, Any
from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.dataset.dataset import UniTrainDataset


class InfiniteTokenBatchedDataset(IterableDataset):
    """
    An infinite iterable dataset that forms batches based on attention token budget.

    Samples are drawn from ``UniTrainDataset`` using weighted multinomial sampling
    across sub-datasets. Each batch accumulates samples until the total attention
    token count approaches ``max_batch_num_attention_tokens``.

    Args:
        dataset: The underlying ``UniTrainDataset``.
        max_batch_num_attention_tokens: Token budget per batch.
        seed: Random seed for reproducibility.
        buffer_size: Number of dataset choices to pre-sample at once.
        num_processes: Total number of distributed processes.
        process_index: Index of the current process.
        skip_samples: Number of samples to skip for training resume.
    """

    def __init__(
        self,
        dataset: UniTrainDataset,
        max_batch_num_attention_tokens: int,
        seed: int = 0,
        buffer_size: int = 10000,
        num_processes: int = 1,
        process_index: int = 0,
        skip_samples: int = 0,
    ):
        self.dataset = dataset
        self.max_batch_num_attention_tokens = max_batch_num_attention_tokens
        self.seed = seed
        self.buffer_size = buffer_size
        self.num_processes = num_processes
        self.process_index = process_index
        self.skip_samples = skip_samples

        self._build_dataset_meta()

    def _build_dataset_meta(self):
        """Build per-dataset sharded index arrays and sampling weights."""
        dataset_weights = []
        self.indices_by_dataset = []

        for i, dataset_info in enumerate(self.dataset.dataset_list):
            start_idx = self.dataset.index_offset[i]
            end_idx = self.dataset.index_offset[i + 1]

            full_indices = torch.arange(start_idx, end_idx, dtype=torch.long)

            # Shard indices across GPUs (e.g. rank 0 gets [0,2,4,...], rank 1 gets [1,3,5,...])
            sharded_indices = full_indices[self.process_index :: self.num_processes]

            self.indices_by_dataset.append(sharded_indices)
            dataset_weights.append(dataset_info.sample_weight)

        self.dataset_weights = torch.tensor(dataset_weights, dtype=torch.float32)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            current_seed = self.seed + worker_id
            num_workers = worker_info.num_workers
            worker_skip_samples = self.skip_samples // num_workers
        else:
            current_seed = self.seed
            worker_skip_samples = self.skip_samples

        generator = torch.Generator()
        generator.manual_seed(current_seed)

        # Initialize per-dataset shuffled queues
        dataset_queues = []
        dataset_pointers = []

        for indices in self.indices_by_dataset:
            shuffled = indices[torch.randperm(len(indices), generator=generator)]
            dataset_queues.append(shuffled)
            dataset_pointers.append(0)

        batch = []
        current_batch_tokens = 0

        choice_buffer = torch.empty(0, dtype=torch.long)
        choice_ptr = 0

        # Fast-forward phase: skip samples for training resume
        if worker_skip_samples > 0:
            samples_skipped = 0
            while samples_skipped < worker_skip_samples:
                if choice_ptr >= len(choice_buffer):
                    choice_buffer = torch.multinomial(
                        self.dataset_weights,
                        self.buffer_size,
                        replacement=True,
                        generator=generator,
                    )
                    choice_ptr = 0

                dataset_idx = choice_buffer[choice_ptr].item()
                choice_ptr += 1

                queue = dataset_queues[dataset_idx]
                ptr = dataset_pointers[dataset_idx]

                if ptr >= len(queue):
                    new_perm = torch.randperm(len(queue), generator=generator)
                    queue = queue[new_perm]
                    dataset_queues[dataset_idx] = queue
                    ptr = 0

                dataset_pointers[dataset_idx] = ptr + 1
                samples_skipped += 1

        # Main sampling loop
        while True:
            # Buffered multinomial sampling for efficiency
            if choice_ptr >= len(choice_buffer):
                choice_buffer = torch.multinomial(
                    self.dataset_weights,
                    self.buffer_size,
                    replacement=True,
                    generator=generator,
                )
                choice_ptr = 0

            dataset_idx = choice_buffer[choice_ptr].item()

            queue = dataset_queues[dataset_idx]
            ptr = dataset_pointers[dataset_idx]

            # Re-shuffle when exhausted
            if ptr >= len(queue):
                new_perm = torch.randperm(len(queue), generator=generator)
                queue = queue[new_perm]
                dataset_queues[dataset_idx] = queue
                ptr = 0

            idx = queue[ptr].item()

            sample = self.dataset[idx]
            if sample is None:
                choice_ptr += 1
                dataset_pointers[dataset_idx] = ptr + 1
                continue

            num_tokens = sample["num_attention_tokens"]
            current_batch_tokens += num_tokens

            # Yield batch when token budget is exceeded.
            # A single oversized sample is always included (never dropped).
            if len(batch) == 0 or current_batch_tokens < self.max_batch_num_attention_tokens:
                batch.append(sample)
                choice_ptr += 1
                dataset_pointers[dataset_idx] = ptr + 1
            else:
                yield batch
                batch = []
                current_batch_tokens = 0


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List]:
    """Collate a list of samples into a batch dict (no stacking, preserves heterogeneous shapes)."""
    inputs_list = []
    labels_list = []
    gen_pixel_values_list = []
    source_pixel_values_list = []
    ref_pixel_values_list = []
    prompts_list = []
    num_attention_tokens_list = []
    num_gen_attention_tokens_list = []
    task_list = []
    dataset_name_list = []
    data_info_list = []

    for sample in batch:
        inputs_list.append(sample["inputs"])
        labels_list.append(sample["labels"])
        gen_pixel_values_list.append(sample["gen_pixel_values"])
        source_pixel_values_list.append(sample["source_pixel_values"])
        ref_pixel_values_list.append(sample["ref_pixel_values"])
        prompts_list.append(sample["prompts"])
        num_attention_tokens_list.append(sample["num_attention_tokens"])
        num_gen_attention_tokens_list.append(sample["num_gen_attention_tokens"])
        task_list.append(sample["task"])
        dataset_name_list.append(sample["dataset_name"])
        data_info_list.append(sample["data_info"])

    return {
        "inputs": inputs_list,
        "labels": labels_list,
        "gen_pixel_values": gen_pixel_values_list,
        "source_pixel_values": source_pixel_values_list,
        "ref_pixel_values": ref_pixel_values_list,
        "prompts": prompts_list,
        "num_attention_tokens": num_attention_tokens_list,
        "num_gen_attention_tokens": num_gen_attention_tokens_list,
        "tasks": task_list,
        "dataset_names": dataset_name_list,
        "data_infos": data_info_list,
    }


def send_to_device(data, device, non_blocking=True):
    """
    Recursively move all tensors in a nested data structure to the specified device.

    Supports torch.Tensor, dict, list, tuple, and other types (returned as-is).
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, Mapping):
        return {k: send_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return type(data)(send_to_device(d, device, non_blocking) for d in data)
    else:
        return data
