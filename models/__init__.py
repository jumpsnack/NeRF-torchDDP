from collections import OrderedDict

import torch

from models.nerf.model import NeRF
from models.nerf.helper import (ndc_rays, chunk_rays, volume_render_radiance_field, sample_pdf_2, get_ray_bundle,
                                mse2psnr)


def build_model(cfg):
    models = OrderedDict()
    models['types'] = list(cfg.models.keys())
    trainable_vars = []
    for _t in models['types']:
        model_params = getattr(cfg.models, _t)

        net = NeRF(num_encoding_fn_xyz=model_params.num_encoding_fn_xyz,
                   include_input_xyz=model_params.include_input_xyz,
                   log_sampling_xyz=model_params.log_sampling_xyz,
                   num_layers=model_params.num_layers,
                   hidden_size=model_params.hidden_size,
                   skip_connect_every=model_params.skip_connect_every,
                   use_viewdirs=model_params.use_viewdirs,
                   num_encoding_fn_dir=model_params.num_encoding_fn_dir,
                   include_input_dir=model_params.include_input_dir,
                   log_sampling_dir=model_params.log_sampling_dir
                   ).to(cfg.gpu)
        model_without_ddp = net
        if cfg.distributed:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[cfg.gpu])
            model_without_ddp = net.module
        trainable_vars += list(model_without_ddp.parameters())

        models[f'net_{_t}'] = net
        models[f'net_{_t}_wo_ddp'] = model_without_ddp
    models['trainable_vars'] = trainable_vars

    return models


def render(hwf, models: OrderedDict, ray_origins, ray_directions, cfg, mode="train"):
    H, W, f = hwf

    viewdirs = None
    if cfg.nerf.use_viewdirs:
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1]
    ]

    if hasattr(cfg.models, "fine"):
        restore_shapes += restore_shapes
    if cfg.dataset.no_ndc is False:
        ray_origins, ray_directions = ndc_rays(H, W, f, 1.0, ray_origins, ray_directions)
    ro = ray_origins.view((-1, 3))
    rd = ray_directions.view((-1, 3))

    near = cfg.dataset.near * torch.ones_like(rd[..., :1])
    far = cfg.dataset.far * torch.ones_like(rd[..., :1])

    rays = torch.cat((ro, rd, near, far), dim=-1)
    if cfg.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    ray_batch = chunk_rays(rays, chunksize=getattr(cfg.nerf, mode).chunksize)
    pred = [
        _render_radiance_field(
            batch,
            models,
            cfg
        )
        for batch in ray_batch
    ]
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if hasattr(cfg.models, "fine"):
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)


def _render_radiance_field(
        ray_batch,
        models: OrderedDict,
        cfg,
        mode="train"
):
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(cfg.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )

    if not getattr(cfg.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(cfg.nerf, mode).num_coarse])

    if getattr(cfg.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = models['net_coarse'](pts, viewdirs)
    (
        rgb_coarse,
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
        white_background=getattr(cfg.nerf, mode).white_background,
    )

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(cfg.nerf, mode).num_fine > 0:
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf_2(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(cfg.nerf, mode).num_fine,
            det=(getattr(cfg.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = models['net_fine'](pts, viewdirs)
        (
            rgb_fine,
            disp_fine,
            acc_fine,
            _,
            _
        ) = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(cfg.nerf, mode).radiance_field_noise_std,
            white_background=getattr(cfg.nerf, mode).white_background,
        )

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
