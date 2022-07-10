import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
from encoders_decoders import *
from ae_templates import sampling
import copy
import random
from tflearn.layers.normalization import batch_normalization
from tensorflow.python.tools import inspect_checkpoint as chip
DATA_DIR=getdata.getdir()
filelist=os.listdir(DATA_DIR)
import sys
from tf_ops.sampling import tf_sampling

from ae_sam import mlp_architecture_ala_iclr_18,movenet
from pointnet_cls import get_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    dist=(dist1 + dist2)/2
    return dist,idx1
def getidpts(pcd,ptid):
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(pcd)[0],dtype=tf.int32),[-1,1,1]),[1,pcd.get_shape()[1].value,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    result=tf.gather_nd(pcd,idx)
    return result
#b*n*1
def CDmax(pcd1,pcd2,d=0.12):
    ptnum1=pcd1.get_shape()[1].value
    ptnum2=pcd2.get_shape()[1].value
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)#b*n
    result=tf.reduce_mean((tf.reduce_max(dist1,axis=1)+tf.reduce_max(dist2,axis=1))/2)
    return result
#data:b*n*3
def get_normal(data,cir=True):
    if not cir:
        result=data
        dmax=np.max(result,axis=1,keepdims=True)
        dmin=np.min(result,axis=1,keepdims=True)
        length=(dmax-dmin)/2
        center=(dmax+dmin)/2
        result=(result-center)/length
    else:
        cen=np.mean(data,axis=1,keepdims=True)
        rdismat=np.sqrt(np.sum(np.square(data-cen),axis=-1))#b*n
        r=np.max(rdismat,axis=-1,keepdims=True) 
        para=1/r
        result=np.expand_dims(para,axis=-1)*(data-cen)#+cen
    return result
def project(pts,data):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pts, data)
    result=getpts(data,idx1,pts.get_shape()[1].value)
    return result
def getpts(pts,idx,knum):
    npoint=idx.get_shape()[1].value
    ptid=tf.reshape(idx,[BATCH_SIZE,-1,1])
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=BATCH_SIZE,dtype=tf.int32),[-1,1,1]),[1,knum,1])
    idx=tf.concat([bid,ptid],axis=-1)
    result=tf.gather_nd(pts,idx)
    return result
def count():
    a=tf.trainable_variables()
    print(a[0].name)
    print(sum([prod(v.get_shape().as_list()) for v in tf.trainable_variables() if not v.name.startswith('E')]))
def resample(data,npts):
    now=data.get_shape()[1].value
    data0=tf.expand_dims(data[:,0,:],axis=1)
    result=tf.concat([data,tf.tile(data0,[1,npts-now,1])],axis=1)
    return result
def train():
    start=0
    num=2048
    n_pc_points=2048
    bsize=32
    k=1
    mlp=[64,128]
    mlp2=[128,128]
    pointcloud_pl=tf.placeholder(tf.float32,[BATCH_SIZE,n_pc_points,3],name='pointcloud_pl')
    outpts=tf.placeholder(tf.float32,[BATCH_SIZE,n_pc_points,3],name='outpts')
    label_pl=tf.placeholder(tf.int32,[None],name='label_pl')
    is_training_pl = tf.placeholder(tf.bool, shape=())
    knum=int(sys.argv[1])
    with tf.variable_scope('sam'):
        samplepts,_=movenet(pointcloud_pl,knum=knum,mlp1=[128,256,256],mlp2=[128,128])
    samplepts=project(samplepts,pointcloud_pl)
    samplepts=resample(samplepts,n_pc_points)

    with tf.variable_scope('ge'):
        pred, end_points = get_model(pointcloud_pl, is_training_pl, bn_decay=None)
        correct = tf.equal(tf.argmax(pred, 1), tf.cast(label_pl,tf.int64))
    filename='cls_result.txt'
    f=open(filename,'a')
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        var=tf.GraphKeys.GLOBAL_VARIABLES
        samvar=tf.get_collection(var,scope='sam')
        istrain=tf.get_collection(var,scope='is_training')
         
        sam_saver=tf.train.Saver(var_list=samvar)
        gevar=tf.get_collection(var,scope='ge')
        ge_saver=tf.train.Saver(var_list=gevar)

        sam_saver.restore(sess, tf.train.latest_checkpoint('./samfiles'))
        ge_saver.restore(sess, tf.train.latest_checkpoint('./pn_cls'))
        sess.run(tf.assign(istrain[0],False))
        testfiles=getdata.getfile(os.path.join(DATA_DIR,'test_files.txt'))

        chamfer_list=[]
        cdm_list=[]
        for i in range(len(testfiles)):
            testdata,label = getdata.load_h5label(os.path.join(DATA_DIR, testfiles[i]))
            testdata=get_normal(testdata,True)
             
            allnum=int(len(testdata)/BATCH_SIZE)*BATCH_SIZE
            batch_num=int(allnum/BATCH_SIZE)
            for batch in range(batch_num):
                start_idx = (batch * BATCH_SIZE) % allnum
                end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
                batch_point=testdata[start_idx:end_idx]
                sampts=sess.run(samplepts,feed_dict={pointcloud_pl:batch_point})
                err_list.append(sess.run(correct,feed_dict={pointcloud_pl:sampts,outpts:batch_point,label_pl:np.squeeze(label[start_idx:end_idx]),is_training_pl:False}))
        cls=np.sum(err_list)/len(BATCH_SIZE*err_list)
        print(cls)
        f.write(str(cls)+'\n')
        f.close()




        
if __name__=='__main__':
    train()
