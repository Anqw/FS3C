#!/bin/bash
#SBATCH -n 80
#SBATCH --gres=gpu:v100:8
#SBATCH --time=48:00:00
#SBATCH --qos=deadline # possible values: short, normal, allgpus
# nvidia-smi
# hostname
# python --version

# sinfo
#module load gcc/6.5.0-fxnktbs
#module load cuda/10.0.130-6rlvsy3
nvcc --version
python setup.py build develop
#python tools/train_net.py --num-gpus 8 --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_base.yaml
#python tools/ckpt_surgery.py --src1 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth --method remove --save-dir checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base
#python tools/train_net.py --num-gpus 8 --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_fc_novel.yaml
#python tools/ckpt_surgery.py --src1 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth --src2 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_fc_novel/model_final.pth --method combine --save-dir checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_combined

python tools/train_net.py --num-gpus 8 --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_base_cosine.yaml
python tools/ckpt_surgery.py --src1 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_nomask_cosine/model_final.pth --method remove --save-dir checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_nomask_cosine
python tools/train_net.py --num-gpus 8 --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_novel.yaml
python tools/ckpt_surgery.py --src1 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_nomask_cosine/model_final.pth --src2 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_fc_novel_cosine/model_final.pth --method combine --save-dir checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined --lvis
python tools/train_net.py --num-gpus 8 --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all.yaml