# Predict pleural effusion using deep learning models

## Prerequisites

```shell
pip install -r requirements.txt
```

## Train

```shell
torchrun --nproc_per_node=4 main.py --mode train --config configs/default_config.py --workdir workdir
```
