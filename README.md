# Predict pleural effusion using deep learning models

## Prerequisites

Create a conda virtual environment by

```shell
conda create -n venv python=3.10 --yes
```

and activate it

```shell
conda activate venv
```

We suggest to install the PyTorch via conda first

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

then install the packages via

```shell
pip install -r requirements.txt
```

## Train

```shell
torchrun --nproc_per_node=[num of GPUs] main.py --mode train --config configs/default_config.py --workdir [name of work directory]
```

Though the NN was implemented using `DistributedDataParallel` in PyTorch, it can be launched on a single GPU while enabling multi-GPU training.

To run on specific GPUs, we can run

```shell
CUDA_VISIBLE_DIVICES=[GPUs] torchrun --nproc_per_node=[num of GPUs] main.py --mode train --config configs/default_config.py --workdir [name of work directory]
```
