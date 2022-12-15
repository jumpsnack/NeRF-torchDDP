import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True):
        super().__init__()

        self.include_input = include_input
        self.periodic_fns = [torch.sin, torch.cos]

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., num_encoding_functions - 1, steps=num_encoding_functions)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** (num_encoding_functions - 1), steps=num_encoding_functions)

    def forward(self, x: torch.Tensor):
        out = [x] if self.include_input else []

        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                out.append(p_fn(x * freq))

        return out[0] if len(out) == 1 else torch.cat(out, dim=-1)
