# Few-Shot Object Detection with Self-Supervising and Cooperative Classifier (Fs3c)
## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Getting Started](#getting-started)

## Installation

Fs3c is built on [FsDet](https://github.com/ucbdrive/few-shot-object-detection).

**Requirements**
* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.3 
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* Dependencies: ```pip install -r requirements.txt```
* pycocotools: ```pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'```
* [fvcore](https://github.com/facebookresearch/fvcore/): ```pip install 'git+https://github.com/facebookresearch/fvcore'``` 
* [OpenCV](https://pypi.org/project/opencv-python/), optional, needed by demo and visualization ```pip install opencv-python```
* GCC >= 4.9

**Build Fs3c**
```angular2html
python setup.py build develop
```
Note: you may need to rebuild Fs3c after reinstalling a different build of PyTorch. 

## Data Preparation
See datasets/README.md for more details.

## Getting Started
###Training & Evaluation
For more detailed instructions on the training procedure, see [TRAIN_INST.md](TRAIN_INST.md).
To evaluate the trained models, run
```angular2html
python tools/test_net.py --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
        --eval-only
```
### Multiple Runs

You can use `tools/run_experiments.py` to do the training and evaluation. For example, to experiment on 30 seeds of the first split of PascalVOC on all shots, run
```angular2html
python tools/run_experiments.py --num-gpus 8 \
        --shots 1 2 3 5 10 --seeds 0 30 --split 1
```

After training and evaluation, you can use `tools/aggregate_seeds.py` to aggregate the results over all the seeds to obtain one set of numbers. To aggregate the 3-shot results of the above command, run
```angular2html
python tools/aggregate_seeds.py --shots 3 --seeds 30 --split 1 \
        --print --plot
```
