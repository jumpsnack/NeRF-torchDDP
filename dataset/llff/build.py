import torch
import numpy as np

from dataset.common import RayInflatedDataset, CollateRay
from dataset.llff.helper import _load_data, normalize, poses_avg, render_path_spiral, recenter_poses, spherify_poses


def _load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75,testskip=1, spherify=False, path_zflat=False):
    poses, bds, imgs = _load_data(basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        # N_views = 120
        N_views = 120 // testskip
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test


def build_llff_data(cfg):
    images, poses, bds, render_poses, i_test = _load_llff_data(cfg.dataset.basedir, cfg.dataset.downsample_factor,
                                                               recenter=True, bd_factor=.75,testskip=cfg.dataset.testskip,
                                                               spherify=False)

    images = torch.Tensor(images)
    poses = torch.Tensor(poses)

    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, cfg.dataset.basedir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if cfg.dataset.llffhold > 0:
        print('Auto LLFF holdout,', cfg.dataset.llffhold)
        i_test = np.arange(images.shape[0])[::cfg.dataset.llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])


    print('DEFINING BOUNDS')
    if cfg.dataset.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.

    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    train_ray_dataset = RayInflatedDataset(hwf, images=images[i_train], poses=poses[i_train])
    test_images = images[i_test]
    test_poses = poses[i_test]
    render_poses = torch.Tensor(render_poses)
    cfg.dataset.near = near
    cfg.dataset.far = far
    collate_fn = CollateRay()

    print(f'Near: {cfg.dataset.near}   Far: {cfg.dataset.far}')

    return train_ray_dataset, test_images, test_poses, render_poses, hwf, collate_fn
