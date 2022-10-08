# PCDNet
[ECCV'22] The codes for Resolution-free Point Cloud Sampling Network with Data Distillation

## Environment
* TensorFlow 1.13.1
* Cuda 10.0
* Python 3.6.9
* numpy 1.14.5

## Dataset
The adopted ShapeNet Part dataset is adopted following [FoldingNet](http://www.merl.com/research/license#FoldingNet), while the ModelNet10 and ModelNet40 datasets follow [PointNet](https://github.com/charlesq34/pointnet.git). Other datasets can also be used. Just revise the path by the (`--filepath`) parameter when training or evaluating the networks.
The files in (`--filepath`) should be organized as

        <filepath>
        ├── <trainfile1>.h5
        ├── <trainfile2>.h5
        ├── ...
        ├── train_files.txt
        └── test_files.txt

where the contents in (`train_files.txt`) or (`test_files.txt`) should include the directory of training or testing h5 files, such as:

        train_files.txt
        ├── <trainfile1>.h5
        ├── <trainfile2>.h5
        ├── ...

## Usage

1. Preparation

```
cd ./tf_ops
bash compile.sh
```

The pre-trained reconstruction and recognition networks are conducted following [S-Net](https://github.com/orendv/learning_to_sample.git). They should be included in the parameter (`--prepath`). 

2. Training

For the reconstruction task,
```
Python3 pc_sampling_rec.py
```

For the recognition task,
```
Python3 pc_sampling_cls.py
```

Note that the path of data (`--filepath`)  should be edited according to your setting.

3. Evaluation

For the reconstruction task,
```
Python3 eva_rec.py
```

For the recognition task,
```
Python3 eva_cls.py
```

The trained weight files should be put in (`--savepath`) to evaluate the sampling performances.

