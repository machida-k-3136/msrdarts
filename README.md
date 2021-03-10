# Minimum Stable Rank Differentiable Architecture Search (MSR-DARTS)
Paper Link
Comming soon

## Requirements
- python 3.6.x
- pytorch 1.4.0
- torchvision 0.5.0
- horovod 0.19.4

## Datasets
CIFAR-10 is downloaded automatically when run ```train_search_cifar.py```.

ImageNet needs to be downloaded manually.

## Architecture search
### Network with 8 layers for CIFAR10
```
python train_search_cifar.py --save save_dir_name --layers 8
```

## Architecture evaluation
### Network with 20 layers for CIFAR10
Train network architecture for CIFAR10 found by MSRDARTS.
```
python train_cifar10.py --auxiliary --cutout
```

### Network with 14 layers for ImageNet
Train network architecture for ImageNet found by MSRDARTS.
```
python train_imagenet_horovod.py --auxiliary
```
It takes about 1day with 32 Tesla P100 GPUs.


## Citation
Commin Soon
