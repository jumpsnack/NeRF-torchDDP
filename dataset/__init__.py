from dataset.llff.build import build_llff_data
from dataset.blender.build import build_blender_data
from dataset.common import *


def build_dataset(cfg):
    if cfg.dataset.type == 'llff':
        return build_llff_data(cfg)
    elif cfg.dataset.type == 'blender':
        return build_blender_data(cfg)
    else:
        raise NotImplementedError(f'Dataset {cfg.dataset.type} is not supported!')
