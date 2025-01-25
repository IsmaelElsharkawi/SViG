# SViG: A Similarity-thresholded Approach for Vision Graph Neural Networks
Accepted in IEEE Access, 2025

## Paper: https://ieeexplore.ieee.org/document/10845790

## Requirements
- Pytorch 1.7.1
- timm 0.3.2
- torchprofile 0.0.4
- apex
- torch_scatter 2.0.7

Please, use the following commands to install the requirements:

`conda env create -f environment.yml`

`pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

`pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.7.1+cu110.html`

## SViG-Ti models pretrained on ImageNet

Each checkpoint has two checkpoints, one for the model updated without using EMA, and an EMA-model. If the EMA column is checked, please use `--model-ema` when evaluating.

|Starting Threshold|Decrement|Top-1|EMA|
|-|-|-|-|
|0.86|0.03|74.6|&cross;|
|0.89|No Decrement|74.1|&check;|
|0.89|0.02|74.2|&check;|
|0.89|0.03|74.6|&cross;|
|0.89|0.04|74.0|&check;|
|0.91|0.03|74.5|&check;|
|0.93|0.03|74.4|&check;|
|0.96|0.03|74.2|&check;|


<!-- Data preparation follows the [official pytorch example](https://github.com/pytorch/examples/tree/main/imagenet) -->

## Evaluation
- Evaluate example:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py <ImageNet_path> --model vig_ti_224_gelu -j 8 --amp --drop 0 --drop-path .1 -b 128 --num-classes 1000 --pretrain_path <checkpoint_path> --evaluate --start-thresh <starting_threshold> --dec <decrement_per_layer> --model-ema(use only if you want to use the EMA checkpoint)
```

## Training

- Training SViG on 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 train.py <path_to_imagenet> --model vig_ti_224_gelu --sched cosine --epochs 300 --opt adamw -j 8 --warmup-lr 1e-6 --mixup .8 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 128 --output <path_to_save_models> --start-thresh <starting_threshold> --dec <decrement_per_layer>
```


## Acknowledgement
This repo partially uses code from [Vision Graph Neural Networks](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch), [deep_gcns_torch](https://github.com/lightaime/deep_gcns_torch) and [timm](https://github.com/rwightman/pytorch-image-models).
