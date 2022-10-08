import h5py
import numpy as np
import os
def getdir():
    BASE_DIR='./'
    DATA_DIR=os.path.join(BASE_DIR,'data')
    return DATA_DIR
def getspdir():
    BASE_DIR='./'
    DATA_DIR=os.path.join(BASE_DIR,'data')
    DATA_DIR=os.path.join(DATA_DIR,'hdf5_data')
    return DATA_DIR
def load_h5(h5_filename):
    f=h5py.File(h5_filename)
    data=f['data'][:]
    return data
def load_h5label(h5_filename):
    f=h5py.File(h5_filename,'r')
    data=f['data'][:]
    label=f['label'][:]
    return data,label
def load_mat(matpath):
    f=scipy.io.loadmat(matpath)
    data=f['data'][:]
    label=f['label'][:]
    return data,label
def getfile(path):
    return [line.strip('\n') for line in open(path)]
