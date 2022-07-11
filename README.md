# PCDNet
The codes for Resolution-free Point Cloud Sampling Network with Data Distillation

## Environment
* TensorFlow 1.13.1
* Cuda 10.0
* Python 3.6.9
* lmdb 0.98  
* tensorpack 0.10.1
* numpy 1.14.5

## Dataset
The adopted ShapeNet Part dataset is adopted following [AE](http://www.merl.com/research/license#FoldingNet), while the ModelNet10 and ModelNet40 datasets follow [PointNet](https://github.com/charlesq34/pointnet.git). Other datasets can also be used.

## Usage

1. Prepartion

```
cd ./tf_ops
bash compile.sh
```

The pre-trained reconstruction and recognition networks are conducted following [S-Net](https://github.com/orendv/learning_to_sample.git).

2. Train

For the reconstruction task,
```
Python3 pc_sampling.py
```

For the recognition task,
```
Python3 pc_sampling_cls.py
```

Note that the paths of data should be edited according to your setting.

3. Test
For the reconstruction task,
```
Python3 vvsam_eva.py
```

For the recognition task,
```
Python3 vvsam_cls.py
```

The trained weight files should be put in (`.\samfiles`) to evaluate the sampling performances.

The pre-trained task and sampling models will come soon.
