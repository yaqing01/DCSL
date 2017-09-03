# DCSL for Person Re-Identification 

Code for paper "Semantics-Aware Deep Correspondence Structure Learning for Robust Person Re-Identification"

## Clone and Installation

Clone the project with Caffe submodule

```sh
$ git clone --recursive https://github.com/yaqing01/DCSL.git
```

[Install Caffe](http://caffe.berkeleyvision.org/installation.html)

## Download and Prepare for the Dataset

1. Download CUHK03/01 datasets from [CUHK Person Re-identification Datasets](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)

2. Unzip the datasets into the dataset/ folder 

The image files are organized as 

```sh
dataset/cuhk03/cuhk03_release/data/campair_1/01_0001_01.jpg
dataset/cuhk03/cuhk03_release/data/campair_1/01_0001_02.jpg
dataset/cuhk03/cuhk03_release/data/campair_1/01_0001_03.jpg
```

3. Generate training data
```sh
cd dataset
python generate_training_pairs.py
cd ..
```

You only need to modify the following configurations in generate_training_pairs.py

```python
set_no = 1 # the training/validation/test split
save_p = 'train_lmdb' # path to save lmdb
dataset_usage = [0,1,0,0,0] # dataset for evaluation
```

## Train the network

We have written all the model templates for training in `models/reid/dcsl`, all we need is to generate the training protos with the specified configurations.

1. Prepare for training
```sh
mkdir experiments
./models/reid/dcsl/prepare.sh
```

2. Train the model using the generated proto files and you can download the pre-trained `bvlc_googlenet.caffemode` for fine-tuning.
```sh
./experiments/reid_dcsl/set01/train_model.sh [GPU-ID]
```

3. Finally, you can evaluate and visualize the trained model by running `code/eval/cuhk03_test.ipynb`

Please cite our work in your publications if it helps your research:

    @inproceedings{ZhangLZZ16,
      author = {Yaqing Zhang and Xi Li and Liming Zhao and Zhongfei Zhang},
      title = {Semantics-Aware Deep Correspondence Structure Learning for Robust Person Re-Identification}, 
      booktitle = {Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, {IJCAI} 2016, New York, NY, USA, 9-15 July 2016},
      pages = {3545--3551},
      year = {2016},
    }
