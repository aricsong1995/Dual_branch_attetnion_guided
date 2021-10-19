# AGCA-Net
This repo contains the training and testing codes for our paper:

### AGCA-Net: Dual-Branch Attention Guided Context Aggregation Network For NonHomogeneous Dehazing


## Prerequisites
- Python >= 3.6  
- [Pytorch](https://pytorch.org/) >= 1.0  
- Torchvision >= 0.2.2  
- Pillow >= 5.1.0  
- Numpy >= 1.14.3
- Scipy >= 1.1.0

## Introduction
- ```train.py``` and ```test.py``` are the entry codes for training and testing the AGCA-Net.
- ```train_loader.py``` and ```test_loader.py``` are used to load the training and validation/testing datasets.
- ```model.py``` defines the model of AGCA-net, and ```Res2net.py``` builds the [Res2net](https://arxiv.org/abs/1904.01169) block.
- ```loss.py``` defines the network for [perceptual loss](https://arxiv.org/abs/1603.08155).
- ```utils.py``` contains all corresponding utilities.
- The ```./trainning_log/``` record the logs.
- The testing hazy images are saved in ```./student_results/```.
- The ```./data/``` folder stores the data for training and testing.

## Quick Start

### 1. Testing
Clone this repo in environment that satisfies the prerequisites
#### Train
```shell
python train.py 
```

#### Test
 ```shell
python test.py 
 ```
## Qualitative Results

<div style="text-align: center">
<img alt="" src="/data/fig1.jpg" style="display: inline-block;" />
</div>

## Citation

If you use any part of this code, please kindly cite




