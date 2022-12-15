from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pe import PositionEmbedding


class NeRF(nn.Module):
    def __init__(self,
                 num_encoding_fn_xyz,
                 include_input_xyz,
                 log_sampling_xyz,
                 num_layers,
                 hidden_size,
                 skip_connect_every,
                 use_viewdirs: bool = False,
                 num_encoding_fn_dir: Optional[int] = None,
                 include_input_dir: Optional[bool] = None,
                 log_sampling_dir: Optional[bool] = None,
                 ):
        super().__init__()

        self.encode_position = PositionEmbedding(
            num_encoding_functions=num_encoding_fn_xyz,
            include_input=include_input_xyz,
            log_sampling=log_sampling_xyz
        )

        self.encode_direction = None
        if use_viewdirs:
            self.encode_direction = PositionEmbedding(
                num_encoding_functions=num_encoding_fn_dir,
                include_input=include_input_dir,
                log_sampling=log_sampling_dir
            )

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every

        self.layers_xyz = torch.nn.ModuleList()
        self.use_viewdirs = use_viewdirs
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_dir, hidden_size))
        for i in range(1, num_layers):
            if i == skip_connect_every:
                self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))
        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2))
        for i in range(num_layers // 2):
            self.layers_dir.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.relu = F.relu

    def forward(self, pts, viewdirs):
        inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        embedded = self.encode_position(inputs_flat)

        if self.encode_direction is not None:
            input_dirs = viewdirs[:, None].expand(pts.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.encode_direction(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        x = embedded

        xyz, dirs = torch.split(x, [self.dim_xyz, self.dim_dir], dim=-1)
        for i, m in enumerate(self.layers_xyz):
            if i == self.skip_connect_every:
                x = m(torch.cat((xyz, x), -1))
            else:
                x = m(x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        else:
            x = self.layers_dir[0](feat)
        x = self.relu(x)
        for m in self.layers_dir[1:]:
            x = m(x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        radiance_field = torch.cat((rgb, alpha), dim=-1)

        radiance_field = torch.reshape(radiance_field, list(pts.shape[:-1]) + [radiance_field.shape[-1]])

        return radiance_field
