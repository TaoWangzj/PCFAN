# Pyramid Channel-based Feature Attention Network for Image Dehazing
Codes for Pyramid Channel-based Feature Attention Network for Image Dehazing.

### Pyramid Channel-based Feature Attention Network for Image Dehazing
[Xiaoqin Zhang](https://scholar.google.com/citations?user=kJCh3k8AAAAJ&hl=en), [Tao Wang](https://taowangzj.github.io/about), Jinxin Wang, Guiying Tang, Li Zhao

Published on _2020 Computer Vision and Image Understanding (CVIU)_

[[Paper](https://taowangzj.github.io/PCFAN/resource/PCFAN.pdf)] [[Project Page](https://taowangzj.github.io/PCFAN/)]
___

## Dependency
- Python >= 3.5  
- [Pytorch](https://pytorch.org/) >= 1.1  
- Torchvision >= 0.4.2  
- Pillow >= 5.1.0  
- Numpy >= 1.14.3
- Scipy >= 1.1.0

## Dataset make
Make you dataset by:
Downloading the ITS training and SOTS testing datasets from [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0).
1. Training dataset: Put hazy and clear folers from downloaded ITS in ```./data/train/ITS/```. 
2. Testing dataset:  put downloaded SOTS (~1000 images) in ```./data/testing/SOTS/```. 
3. Note: train.txt and val.txt provide the image list for training and testing, respectively.

## Code Introduction
- ```train.py``` and ```test.py``` are the codes for training and testing the PCFAN.
- ```./datasets/datasets.py``` is used to load the training and testing datasets.
- ```./model/network.py``` defines the structure of PCFAN.
- ```./loss/edg_loss.py``` defines the proposed Edge loss.
- ```utils.py``` contains all utilities used for training and testing the PCFAN.
- ```./checkpoints/indoor_haze_best.pth``` and ```./checkpoints/outdoor_haze.pth``` are the trained weights for indoor and outdoor in SOTS from [RESIDE](https://www.baidu.com).
- The ```./logs/indoor_log.log``` and ```./logs/outdoor_log.log``` record the core logs.
- The ```./logs/run_indoor.log ``` and ``` ./logs/run_outdoor.log``` record the detailed training logs.
- The testing hazy images are saved in ```./results/indoor_results/``` and ```./results/outdoor_results/```, respectively.
- The ```./data/``` folder stores the training and testing data.

## Train
You can train the model for indoor dataset by:
```
python  train.py  --nEpochs 200 --category indoor 
```
You can train the model for outdoor dataset by:
```
python  train.py  --nEpochs 10 --category outdoor 
```

## Test
You can test you model on indoor of SOTS [dataset](https://sites.google.com/view/reside-dehaze-datasets/reside-v0).
```
python test.py --category indoor 
```

You can test you model on outdoor of SOTS [dataset](https://sites.google.com/view/reside-dehaze-datasets/reside-v0).
```
python test.py --category outdoor 
```
## Refenrece:
```
@article{Zhang2020pyramid,
title = {Pyramid Channel-based Feature Attention Network for image dehazing},
author = {Zhang, Xiaoqin and Wang, Tao and Wang, Jinxin and Tang, Guiying  and Zhao, Li},
journal = {Computer Vision and Image Understanding},
volume = {197-198},
pages = {103003},
year = {2020},
publisher={Elsevier}
}
```

