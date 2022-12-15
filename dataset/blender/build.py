import os
import torch
import numpy as np
import imageio
import json
import cv2

from dataset.blender.helper import pose_spherical
from dataset.common import CollateRay, RayInflatedDataset


def _load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split


def build_blender_data(cfg):
    images, poses, render_poses, hwf, i_split = _load_blender_data(cfg.dataset.basedir, cfg.dataset.half_res,
                                                                   cfg.dataset.testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, cfg.dataset.basedir)
    i_train, i_val, i_test = i_split

    images = torch.Tensor(images)
    poses = torch.Tensor(poses)

    if cfg.nerf.train.white_background:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    train_ray_dataset = RayInflatedDataset(hwf, images=images[i_train], poses=poses[i_train, :3, :4])
    render_poses = torch.Tensor(render_poses)
    test_images = images[i_test]
    test_poses = poses[i_test]
    collate_fn = CollateRay()

    print(f'Near: {cfg.dataset.near}   Far: {cfg.dataset.far}')

    return train_ray_dataset, test_images, test_poses, render_poses, hwf, collate_fn
