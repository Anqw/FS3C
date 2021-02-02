cd datasets
mkdir lvissplit
cd lvissplit
wget http://dl.yf.io/fs-det/datasets/lvissplit/lvis_shots.json
cd -

mkdir coco
# download images
cd coco

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && rm val2017.zip
cd -

mkdir lvis
cd lvis
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
unzip lvis_v1_train.json.zip && rm lvis_v1_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
unzip lvis_v1_val.json.zip && rm lvis_v1_val.json.zip
cd -
#srun singularity exec --nv fs3c.sif python split_lvis_annotation.py
#python split_lvis_annotation.py