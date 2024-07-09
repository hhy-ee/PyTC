import os
import argparse

import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

__all__ = [
    'init_devices',
]


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_devices(args, cfg):
    if args.distributed:  # parameters to initialize the process group
        assert torch.cuda.is_available(), \
            "Distributed training without GPUs is not supported!"

        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK",
                        "LOCAL_RANK", "WORLD_SIZE")}
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        dist.init_process_group(cfg.SYSTEM.DISTRIBUTED_BACKEND, init_method='env://')
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )

        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        n = torch.cuda.device_count() // args.local_world_size
        device_ids = list(
            range(args.local_rank * n, (args.local_rank + 1) * n))

        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

        print(
            f"[{os.getpid()}] rank = {dist.get_rank()} ({args.rank}), "
            + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
        )

        manual_seed = args.local_rank if args.manual_seed is None \
            else args.manual_seed
    else:
        manual_seed = 0 if args.manual_seed is None else args.manual_seed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("rank: {}, device: {}, seed: {}".format(args.local_rank, device, manual_seed))
    # use manual_seed seeds for reproducibility
    init_seed(manual_seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    return device
