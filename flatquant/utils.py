import random
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import transformers

import logging

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from torch.distributed.device_mesh import init_device_mesh

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


@dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    ddp_size: int
    fsdp_size: int
    dp_rank: int
    fsdp_rank: int
    dp_group: Optional[dist.ProcessGroup]
    fsdp_group: Optional[dist.ProcessGroup]
    device_mesh: Optional[object]

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def skip_initialization():
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

def cleanup_memory(verbose=True) -> None:
    """Clear GPU memory by running garbage collection and emptying cache."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbose:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def _infer_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device('cuda', local_rank)
    return torch.device('cpu')


def init_distributed(args) -> Optional[DistEnv]:
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return None

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    device = _infer_device(local_rank)

    ddp_size = args.ddp_size
    fsdp_size = args.fsdp_size
    assert ddp_size * fsdp_size == world_size, "ddp_size ({ddp_size}) * fsdp_size ({fsdp_size}) must match WORLD_SIZE ({world_size})."

    dp_rank = rank // fsdp_size
    fsdp_rank = rank % fsdp_size

    fsdp_group = None
    if fsdp_size > 1:
        fsdp_groups = []
        for dp_idx in range(ddp_size):
            ranks = [dp_idx * fsdp_size + shard_idx for shard_idx in range(fsdp_size)]
            fsdp_groups.append(dist.new_group(ranks=ranks))
        fsdp_group = fsdp_groups[dp_rank]

    dp_group = None
    if ddp_size > 1:
        dp_groups = []
        for shard_idx in range(fsdp_size):
            ranks = [shard_idx + fsdp_size * dp_idx for dp_idx in range(ddp_size)]
            dp_groups.append(dist.new_group(ranks=ranks))
        dp_group = dp_groups[fsdp_rank]

    device_mesh = None
    if ddp_size > 1 or fsdp_size > 1:
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(ddp_size, fsdp_size),
            mesh_dim_names=("dp", "fsdp"),
        )

    return DistEnv(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        ddp_size=ddp_size,
        fsdp_size=fsdp_size,
        dp_rank=dp_rank,
        fsdp_rank=fsdp_rank,
        dp_group=dp_group,
        fsdp_group=fsdp_group,
        device_mesh=device_mesh,
    )


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2/3/Qwen-2."""
    no_split_module_classes = ['LlamaDecoderLayer', 'Qwen2DecoderLayer']
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split_module_classes)

    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)

    dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=model.state_dict())
    cleanup_memory()


def seed_everything(seed=0) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    transformers.set_seed(seed)
