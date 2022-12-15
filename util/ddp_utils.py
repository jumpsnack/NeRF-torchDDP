import os
import torch
import torch.distributed as dist
import numpy as np
from contextlib import contextmanager

def init_distributed_mode_ddp(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.gpu = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    # torch.distributed.init_process_group(backend=args.dist_backend,timeout=datetime.timedelta(seconds=5400),
                                         world_size=args.world_size, rank=args.local_rank)
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)
    args.experiment.train_iters = args.experiment.train_iters // get_world_size()
    args.scheduler.lr_decay = args.scheduler.lr_decay // get_world_size()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_seed(seed):
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    # Decorator to make all processes in distributed training wait for each local_master to do something
    if not is_dist_avail_and_initialized():
        yield
    else:
        if local_rank not in [-1, 0]:
            # dist.barrier(device_ids=[local_rank])
            dist.barrier()
        yield
        if local_rank == 0:
            # dist.barrier(device_ids=[0])
            dist.barrier()

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)