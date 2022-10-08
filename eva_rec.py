import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
from encoders_decoders import *
from tf_ops.CD import tf_nndistance 
from builtins import object
from tf_ops.grouping.tf_grouping import group_point, knn_point
import copy
import random
from tflearn.layers.normalization import batch_normalization
from tensorflow.python.tools import inspect_checkpoint as chip
import time
import sys
import argparse

from tf_ops.sampling import tf_sampling

from ae_sam import mlp_architecture_ala_iclr_18,movenet
from pointnet_cls import get_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    #dist=tf.reduce_mean(tf.maximum(tf.sqrt(dist1),tf.sqrt(dist2)))
    dist=(dist1 + dist2)/2
    return dist,idx1
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
        length=np.max((dmax-dmin)/2,axis=-1,keepdims=True)
        center=(dmax+dmin)/2
        result=(result-center)/length
    else:
        cen=np.mean(data,axis=1,keepdims=True)
        rdismat=np.sqrt(np.sum(np.square(data-cen),axis=-1))#b*n
        r=np.max(rdismat,axis=-1,keepdims=True)
        para=1/r
        #print(np.shape(para))
        para=np.expand_dims(para,axis=-1)
        result=para*(data-cen)#+cen
    return result,para,cen
def project(pts,data,bsize):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pts, data)
    result=getpts(data,idx1,pts.get_shape()[1].value,bsize)
    return result
def getpts(pts,idx,knum, bsize):
    npoint=idx.get_shape()[1].value
    ptid=tf.reshape(idx,[bsize,-1,1])#tf.cast(tf.tile(tf.reshape(idx,[-1,npoint,1]),[BATCH_SIZE,1,1]),tf.int32)
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=bsize,dtype=tf.int32),[-1,1,1]),[1,knum,1])
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
def restore_into_scope(model_path, scope_name, sess):
    
    global_vars = tf.global_variables()
    tensors_to_load = [v for v in global_vars if v.name.startswith(scope_name + "/")]

    load_dict = {}
    for j in range(0, np.size(tensors_to_load)):
        tensor_name = tensors_to_load[j].name
        #print(tensor_name)
        tensor_name = tensor_name[0:-2]  # remove ':0'
        tensor_name = tensor_name.replace(scope_name + "/", "sam/")  # remove scope
        #print(tensor_name)
        load_dict.update({tensor_name: tensors_to_load[j]})
    #print(load_dict)
    loader = tf.train.Saver(var_list=load_dict)
    loader.restore(sess, model_path)
    print(
        "Model restored from: {0} into scope: {1}.".format(model_path, scope_name)
    )


def evaluate(args):
    pointcloud_pl=tf.placeholder(tf.float32,[None,args.ptnum,3],name='pointcloud_pl')
    outpts=tf.placeholder(tf.float32,[None,args.ptnum,3],name='outpts')
    label_pl=tf.placeholder(tf.int32,[None],name='label_pl')
    is_training_pl = tf.placeholder(tf.bool, shape=())
    
    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(args.ptnum, 128, dnum=3,mode='fc')
    with tf.variable_scope('ge'):
        word=encoder(pointcloud_pl,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],strides=enc_args['strides'],b_norm=enc_args['b_norm'],b_norm_decay=1.0,verbose=enc_args['verbose'])
        out=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_decay=1.0,b_norm_finish=dec_args['b_norm_finish'],b_norm_decay_finish=1.0,verbose=dec_args['verbose'])
    out=tf.reshape(out,[-1,args.ptnum,3])

    count()
    with tf.variable_scope('sam'):
        samplepts,_=movenet(pointcloud_pl,knum=args.knum,mlp1=[128,256,256],mlp2=[128,128],startcen=None,infer=True) 

    samplepts=project(samplepts,pointcloud_pl,args.batch_size)
    samplepts=resample(samplepts,args.ptnum)

    chamferloss=chamfer_big(outpts,out)[0]#+chamfer_big(outpts,fpsout)[0]
    cdmloss=CDmax(outpts,out)

    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        var=tf.GraphKeys.GLOBAL_VARIABLES
        samvar=tf.get_collection(var,scope='sam')
        
        sam_saver=tf.train.Saver(var_list=samvar)
        gevar=tf.get_collection(var,scope='ge')
        ge_saver=tf.train.Saver(var_list=gevar)

        from tflearn import is_training
        is_training(False, session=sess) 

        if os.path.exists(os.path.join(args.savepath,'checkpoint')):
            sam_saver.restore(sess, tf.train.latest_checkpoint(args.savepath))
        else:
            print('There is no trained checkpoint!')
            assert False

        if os.path.exists(os.path.join(args.prepath,'checkpoint')):
            ge_saver.restore(sess, tf.train.latest_checkpoint(args.prepath))
        else:
            print('There is no pretrained checkpoint!')
            assert False

        testfiles=getdata.getfile(os.path.join(args.filepath,'test_files.txt'))

        chamfer_list=[]
        cdm_list=[]
        f=open('rec_result.txt','a')
        for i in range(len(testfiles)):
            testdata = getdata.load_h5(os.path.join(args.filepath, testfiles[i]))
            testdata,_,_=get_normal(testdata,True)#[:,:,:3]
            testdata=testdata[:,:,:3]
            
            allnum=int(len(testdata)/args.batch_size)*args.batch_size
            batch_num=int(allnum/args.batch_size)

            for batch in range(batch_num):
                start_idx = (batch * args.batch_size) % allnum
                end_idx=(batch*args.batch_size)%allnum+args.batch_size

                batch_point=testdata[start_idx:end_idx]

                sampts=sess.run(samplepts,feed_dict={pointcloud_pl:batch_point})

                chamfer_list.append(sess.run(chamferloss,feed_dict={pointcloud_pl:sampts,outpts:batch_point}))
                cdm_list.append(sess.run(cdmloss,feed_dict={pointcloud_pl:sampts,outpts:batch_point}))
                
        print(mean(chamfer_list),mean(cdm_list))
        f.write(str(mean(chamfer_list))+','+str(mean(cdm_list))+'\n')
        f.close() 

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--ptnum', type=int, default=2048, help='The number of points')
    parser.add_argument('--knum', type=int, default=32, help='The number of sampled points')
    parser.add_argument('--filepath', type=str, default='./data', help='The path of test data')
    parser.add_argument('--savepath', type=str, default='./modelvv_sam/', help='The path of saved checkpoint')
    parser.add_argument('--prepath', type=str, default='./ae_cd/', help='The path of saved checkpoint')
 
    args=parser.parse_args()
    evaluate(args)
