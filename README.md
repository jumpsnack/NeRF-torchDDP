# NeRF-torchDDP


This repo is an unofficial implementation of ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/pdf/2003.08934.pdf). The codebase is implemented using [PyTorch](https://pytorch.org/) and tested on [Ubuntu](https://ubuntu.com/) 20.04.4 LTS.

## Prerequisite

### `Configure environment`

Install [Anaconda](https://www.anaconda.com/).

Create and activate a virtual environment.

    conda create --name nerf-torchddp python=3.8
    conda activate nerf-torchddp

The code is tested with python 3.8, cuda == 11.1, pytorch == 1.10.1.
Install the required additional packages.

    pip install -r requirements.txt

### `Download dataset`

All datasets must be downloaded to a directory `../data` and must follow the below organization. 
```bash
├──data/
    ├──nerf_synthetic/
    ├──nerf_llff_data/
├──NeRF-DDP/
    ├──main.py/
    ...
```

We refer to IBRNet's repository to download and prepare data.
```bash
# Blender dataset
gdown https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
unzip nerf_synthetic.zip

# LLFF dataset
gdown https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
unzip nerf_llff_data.zip
```

## Usage
### `Training`

```bash
# Single GPU
python main.py --config config/llff.yml --scene fern
python main.py --config config/blender.yml --scene lego

# Multi 2GPUs
python -m torch.distributed.launch --nproc_per_node 2 main.py --config config/llff.yml --scene fern
```

## Acknowledgement

The implementation took reference from [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [krrish94/nerf-pytorch](https://github.com/krrish94/nerf-pytorch). I thank the authors for their generosity to release code.