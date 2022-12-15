from models import get_ray_bundle
from torch.utils.data import Dataset, DataLoader
import torch


class RayDataset(Dataset):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses


class RayInflatedDataset(RayDataset):
    def __init__(self, hwf, **kwargs):
        super().__init__(**kwargs)
        rays = torch.stack([torch.stack(get_ray_bundle(hwf, p), 0) for p in self.poses[:, :3, :4]],
                           0)  # [N, ro+rd, H, W, 3]
        rays_rgb = torch.cat([rays, self.images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = torch.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        self.rays_rgb = rays_rgb.float()

    def __len__(self):
        return len(self.rays_rgb)

    def __getitem__(self, idx):
        return self.rays_rgb[idx]


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()
        self.seed = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            if hasattr(self.sampler, 'set_epoch'):
                self.seed += 1
                self.sampler.set_epoch(self.seed)
            batch = next(self.dataset_iterator)
        return batch


class CollateRay:
    def __call__(self, batch):
        batch = torch.stack(batch).transpose(0, 1)
        return batch[:2], batch[2]
