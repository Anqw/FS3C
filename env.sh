conda create -n fsod python=3.6
conda activate fsod
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install opencv-python

