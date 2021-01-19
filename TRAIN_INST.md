# FS3C Training Instructions

FS3C is trained in two stages. 

## Stage 1: Base Training

First train a base model. To train a base model on the first split of PASCAL VOC, run
```angular2html
python tools/train_net.py --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_base1.yaml
```

## Stage 2: Few-Shot Fine-Tuning

### Initialization

After training the base model, run ```tools/ckpt_surgery.py``` to obtain an initialization for the full model.

#### Random Weights

To randomly initialize the weights corresponding to the novel classes, run
```angular2html
python tools/ckpt_surgery.py \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1
```
The resulting weights will be saved to `checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery.pth`.

#### Novel Weights

To use novel weights, fine-tune a predictor on the novel set. We reuse the base model trained in the previous stage but retrain the last layer from scratch. First remove the last layer from the weights file by running
```angular2html
python tools/ckpt_surgery.py \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method remove \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1
```

Next, fine-tune the predictor on the novel set by running
```angular2html
python tools/train_net.py --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_novel1_1shot.yaml 
```

Finally, combine the base weights from the base model with the novel weights by running
```angular2html
python tools/ckpt_surgery.py \
        --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --src2 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel1_1shot/model_final.pth \
        --method combine \
        --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1
```
The resulting weights will be saved to `checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_combine.pth`.

### Fine-Tuning

We will then fine-tune the last layer of the full model on a balanced dataset by running
```angular2html
python tools/train_net.py --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml
```
