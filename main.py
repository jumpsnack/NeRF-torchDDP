import warnings

warnings.filterwarnings(action='ignore')

import argparse
import torch
import os
import yaml

import torch.nn.functional as F
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
import torch.distributed as dist
from timm.utils import NativeScaler
from tqdm import tqdm

from models import build_model, render, get_ray_bundle, mse2psnr
from dataset import build_dataset, InfiniteDataLoader
from util import (CfgNode,
                  init_distributed_mode_ddp, init_seed, get_world_size, get_rank, is_main_process,
                  plot_video, plot_image, colorize_np)


def main(cfg):
    init_distributed_mode_ddp(cfg)
    init_seed(cfg.experiment.seed)

    train_dataset, test_images, test_poses, render_poses, hwf, collate_fn = build_dataset(cfg)
    if cfg.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_train = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = RandomSampler(train_dataset)
    train_ray_dataloader = InfiniteDataLoader(train_dataset, batch_size=cfg.nerf.train.num_random_rays,
                                              sampler=sampler_train,
                                              num_workers=cfg.experiment.num_workers, collate_fn=collate_fn,
                                              pin_memory=True, drop_last=True)
    if is_main_process():
        render_poses, test_images, test_poses = render_poses.cuda(), test_images.cuda(), test_poses.cuda()

    net_nerf = build_model(cfg)
    optimizer = getattr(torch.optim, cfg.optimizer.type)(net_nerf['trainable_vars'],
                                                         lr=cfg.optimizer.lr)
    loss_scaler = NativeScaler()
    best_psnr = 0.
    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location='cpu')
        start_step = ckpt['global_step'] + 1
        best_psnr = ckpt['best_psnr']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        loss_scaler.load_state_dict(ckpt['loss_scaler_state_dict'])
        net_nerf['net_coarse_wo_ddp'].load_state_dict(ckpt['net_coarse_state_dict'])
        if net_nerf.get('net_fine_wo_ddp') and ckpt['net_fine_state_dict']:
            net_nerf['net_fine_wo_ddp'].load_state_dict(ckpt['net_fine_state_dict'])
        print('=' * 7, f'resume from {cfg.resume}', '=' * 7)
        print(f'global_step = {start_step}   Previous best psnr = {best_psnr}')

    train_iter = iter(train_ray_dataloader)
    pbar = tqdm(range(cfg.experiment.train_iters), disable=not is_main_process())
    pbar.update(start_step - 1)

    for global_step in range(start_step, cfg.experiment.train_iters):
        pbar.update()

        train_stats = train_one_step(net_nerf, hwf, train_iter, optimizer, loss_scaler, global_step)

        pbar.set_description(
            f'lr: {train_stats["train/lr"]:.6f}  '
            f'loss: {train_stats["train/loss"]:.4f}  '
            f'loss0: {train_stats["train/loss0"]:.4f}  '
            f'psnr: {train_stats["train/psnr"]:.1f}  '
            f'psnr0: {train_stats["train/psnr0"]:.1f}'
        )

        if is_main_process() and global_step % cfg.experiment.validate_every == 0 and global_step > 0:
            test_stats = evaluate(global_step, net_nerf, test_images, test_poses, hwf)
            save_dict = {
                'global_step': global_step,
                'best_psnr': best_psnr,
                'net_coarse_state_dict': net_nerf['net_coarse_wo_ddp'].state_dict(),
                'net_fine_state_dict': net_nerf['net_fine_wo_ddp'].state_dict() if net_nerf.get(
                    'net_fine_wo_ddp') else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_scaler_state_dict': loss_scaler.state_dict()
            }

            if test_stats['val/psnr'] > best_psnr:
                best_psnr = test_stats['val/psnr']
                save_dict['best_psnr'] = best_psnr
                torch.save(save_dict, os.path.join(cfg.experiment.logdir, 'best.tar'))
                print(f"saved checkpoint at {os.path.join(cfg.experiment.logdir, 'best.tar')}")
            torch.save(save_dict, os.path.join(cfg.experiment.logdir, 'last.tar'))
            print(f"saved checkpoint at {os.path.join(cfg.experiment.logdir, 'last.tar')}")

            print(
                f"[VAL] loss: {test_stats['val/loss']:.4f}  psnr: {test_stats['val/psnr']:.2f}  Best psnr: {best_psnr:.2f}")


        if is_main_process() and global_step % cfg.experiment.plot_every == 0 and global_step > 0:
            plot_render(global_step, net_nerf, render_poses, hwf)

        if cfg.distributed:
            dist.barrier()

    if is_main_process():
        test_stats = evaluate(global_step, net_nerf, test_images, test_poses, hwf)
        save_dict = {
            'global_step': global_step,
            'best_psnr': best_psnr,
            'net_coarse_state_dict': net_nerf['net_coarse_wo_ddp'].state_dict(),
            'net_fine_state_dict': net_nerf['net_fine_wo_ddp'].state_dict() if net_nerf.get(
                'net_fine_wo_ddp') else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_scaler_state_dict': loss_scaler.state_dict()
        }

        if test_stats['val/psnr'] > best_psnr:
            best_psnr = test_stats['val/psnr']
            save_dict['best_psnr'] = best_psnr
            torch.save(save_dict, os.path.join(cfg.experiment.logdir, 'best.tar'))
            print(f"saved checkpoint at {os.path.join(cfg.experiment.logdir, 'best.tar')}")
        torch.save(save_dict, os.path.join(cfg.experiment.logdir, 'last.tar'))
        print(f"saved checkpoint at {os.path.join(cfg.experiment.logdir, 'last.tar')}")

        print(
            f"[VAL] loss: {test_stats['val/loss']:.4f}  psnr: {test_stats['val/psnr']:.2f}  Best psnr: {best_psnr:.2f}")

        plot_render(global_step, net_nerf, render_poses, hwf)

    if cfg.distributed:
        dist.barrier()

def train_one_step(models, hwf, train_iter, optimizer, loss_scaler, global_step):
    [models[f'net_{_t}'].train() for _t in models['types']]

    batch = next(train_iter)
    batch_rays, target_c = [t.cuda() for t in batch]
    ray_origins, ray_directions = batch_rays

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        rgb_coarse, _, _, rgb_fine, _, _ = render(hwf,
                                                  models,
                                                  ray_origins,
                                                  ray_directions,
                                                  cfg,
                                                  mode="train")

        coarse_loss = F.mse_loss(rgb_coarse[..., :3], target_c[..., :3])
        coarse_psnr = mse2psnr(coarse_loss.item())

        if rgb_fine is not None:
            fine_loss = F.mse_loss(rgb_fine[..., :3], target_c[..., :3])
            fine_psnr = mse2psnr(fine_loss.item())

    loss = coarse_loss + fine_loss if rgb_fine is not None else 0
    loss_scaler._scaler.scale(loss).backward()
    loss_scaler._scaler.step(optimizer)
    loss_scaler._scaler.update()

    num_decay_steps = cfg.scheduler.lr_decay * 1000
    lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (global_step / num_decay_steps)
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_new

    return {
        'train/lr': lr_new,
        'train/loss': coarse_loss.item(),
        'train/loss0': fine_loss.item() if rgb_fine is not None else 0,
        'train/psnr': coarse_psnr,
        'train/psnr0': fine_psnr if rgb_fine is not None else 0
    }


@torch.no_grad()
def evaluate(i, models, test_images, test_poses, hwf):
    [models[f'net_{_t}'].eval() for _t in models['types']]

    testdir = os.path.join(cfg.experiment.logdir, 'test_imgs_{:06d}'.format(i))
    os.makedirs(testdir, exist_ok=True)
    losses, psnrs = [], []
    for s, (p, c) in enumerate(tqdm(zip(test_poses, test_images), total=len(test_poses))):
        ray_origins, ray_directions = get_ray_bundle(hwf, p[:3, :4])
        with torch.cuda.amp.autocast():
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _ = render(hwf,
                                                                        models,
                                                                        ray_origins,
                                                                        ray_directions,
                                                                        cfg,
                                                                        mode="validation")
            loss = F.mse_loss(rgb_coarse[..., :3], c[..., :3])
            if rgb_fine is not None:
                loss = F.mse_loss(rgb_fine[..., :3], c[..., :3])
            psnr = mse2psnr(loss.item())
            losses.append(loss.item())
            psnrs.append(psnr)
            plot_rgb = rgb_coarse if rgb_fine is None else rgb_fine
            plot_disp = disp_coarse if disp_fine is None else disp_fine
            plot_image(plot_rgb.cpu().numpy(), os.path.join(testdir, '{:03d}.png'.format(s)))
            plot_image(colorize_np(np.nan_to_num(plot_disp.cpu().numpy())), os.path.join(testdir, '{:03d}_disp.png'.format(s)))
    total_loss, total_psnr = np.mean(losses), np.mean(psnrs)
    return {'val/loss': total_loss, 'val/psnr': total_psnr}


@torch.no_grad()
def plot_render(i, models, render_poses, hwf):
    [models[f'net_{_t}'].eval() for _t in models['types']]

    renderdir = os.path.join(cfg.experiment.logdir, 'render_imgs_{:06d}'.format(i))
    os.makedirs(renderdir, exist_ok=True)
    rgbs, disps = [], []
    for s, p in enumerate(tqdm(render_poses, total=len(render_poses), desc='[Plot render]')):
        ray_origins, ray_directions = get_ray_bundle(hwf, p[:3, :4])
        with torch.cuda.amp.autocast():
            _, _, _, rgb_fine, disp_fine, _ = render(hwf, models, ray_origins, ray_directions, cfg,
                                                     mode="validation")
            rgb_fine = rgb_fine.cpu().numpy()
            disp_fine = disp_fine.cpu().numpy()
            rgbs.append(rgb_fine)
            disps.append(colorize_np(np.nan_to_num(disp_fine)))
            plot_image(disps[-1], os.path.join(renderdir, 'spiral_{:03d}_disp.png'.format(s)))
            plot_image(rgbs[-1], os.path.join(renderdir, 'spiral_{:03d}.png'.format(s)))
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    moviebase = os.path.join(renderdir, 'spiral_{:06d}_'.format(i))
    plot_video(rgbs, moviebase + 'rgb.mp4')
    plot_video(disps, moviebase + 'disp.mp4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='config/{blender, llff}.yml')
    parser.add_argument("--scene", type=str, required=True, help='lego, fern, drums ...')
    parser.add_argument("--resume", type=str, default='')

    parser.add_argument('--local_rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg_dict.update(args.__dict__)
        cfg = CfgNode(cfg_dict)

        cfg.dataset.scene = args.scene
        cfg.dataset.basedir = os.path.join(cfg.dataset.basedir, args.scene)
        cfg.experiment.logdir = os.path.join(cfg.experiment.logdir, f'{cfg.experiment.id}_{args.scene}')

    main(cfg)
