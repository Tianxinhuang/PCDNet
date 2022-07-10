import tensorflow as tf
from numpy import *
import os
import getdata
#import tf_util
import copy
import random
import point_choose
import numpy as np

from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
from provider import shuffle_data,shuffle_points,rotate_point_cloud,jitter_point_cloud

import sys
from pointnet_cls import get_model,get_loss
#import tf_util 

DATA_DIR=getdata.getdir()
filelist=os.listdir(DATA_DIR)

trainfiles=getdata.getfile(os.path.join(DATA_DIR,'train_files.txt'))
testfiles=getdata.getfile(os.path.join(DATA_DIR,'test_files.txt'))

EPOCH_ITER_TIME=500
BATCH_ITER_TIME=5000
BASE_LEARNING_RATE=0.001
REGULARIZATION_RATE=0.00001
BATCH_SIZE=16
DECAY_STEP=1000*BATCH_SIZE
DECAY_RATE=0.7

NUM_CLASSES = 40
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

PT_NUM=2048
FILE_NUM=1
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
def train_one_epoch(sess,ops,train_writer):
    for j in range(FILE_NUM):
        traindata,label = getdata.load_h5label(os.path.join(DATA_DIR, trainfiles[j]))
        #traindata,label=getdata.load_mat(os.path.join(os.path.join(getdata.getdir(),'ModelNet10'),'train.mat'))
        #label=reshape(label-1,[-1,1])
        #traindata=get_normal(traindata)

        label=squeeze(label)
        traindata,label,_=shuffle_data(traindata,label)
        traindata=shuffle_points(traindata)
        traindata=traindata[:,:PT_NUM]
        traindata=rotate_point_cloud(traindata)
        traindata=jitter_point_cloud(traindata)

        allnum=int(len(traindata)/BATCH_SIZE)*BATCH_SIZE
        batch_num=int(allnum/BATCH_SIZE)
        for batch in range(batch_num):
            start_idx = (batch * BATCH_SIZE) % allnum
            end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
            batch_point = traindata[start_idx:end_idx]
            feed_dict = {ops['pointcloud_pl']: batch_point,ops['label_pl']:label[start_idx:end_idx],ops['is_training_pl']:True}
            resi = sess.run([ops['trainstep'],ops['loss'],ops['accuracy'],ops['zhengze']], feed_dict=feed_dict)
            if batch % 16 == 0:
                result = sess.run(ops['merged'], feed_dict=feed_dict)
                train_writer.add_summary(result, batch)
                print('epoch: %d '%ops['epoch'],'file: %d '%j,'batch: %d' %batch)
                print('batch_accuracy: %f'%resi[2])
                print('loss: ',[resi[1]])
                print('regularization: ', resi[-1])
def eval_one_epoch(sess,ops):
    acculist=[]
    for j in range(2):
        traindata,label = getdata.load_h5label(os.path.join(DATA_DIR, testfiles[j]))
        traindata=traindata[:,:PT_NUM]
        
        label=squeeze(label)
        allnum=int(len(traindata)/BATCH_SIZE)*BATCH_SIZE
        batch_num=int(allnum/BATCH_SIZE)
        for batch in range(batch_num):
            start_idx = (batch * BATCH_SIZE) % allnum
            end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
            batch_point = traindata[start_idx:end_idx]
            feed_dict = {ops['pointcloud_pl']: batch_point,ops['label_pl']:label[start_idx:end_idx],ops['is_training_pl']:False}
            resi = sess.run([ops['trainstep'],ops['loss'],ops['accuracy']], feed_dict=feed_dict)
            acculist.append(resi[2])
            if batch % 16 == 0:
                print('file: %d '%j,'batch: %d' %batch)
                print('batch_accuracy: %f'%resi[2])
                print('loss: ',resi[1])
    print('mean accuarcy: %f'%mean(acculist))

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    pointcloud_pl=tf.placeholder(tf.float32,[BATCH_SIZE,PT_NUM,3],name='pointcloud_pl')
    label_pl=tf.placeholder(tf.int32,[None],name='label_pl')
    is_training_pl = tf.placeholder(tf.bool, shape=())

    global_step=tf.Variable(0,trainable=False)
    bn_decay=get_bn_decay(global_step*BATCH_SIZE)

    with tf.variable_scope('ge'): 
        pred, end_points = get_model(pointcloud_pl, is_training_pl, bn_decay=bn_decay)
        loss = get_loss(pred, label_pl, end_points)
    correct = tf.equal(tf.argmax(pred, 1), tf.cast(label_pl,tf.int64))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.cast(tf.shape(label_pl)[0],tf.float32) 

    zhengze=tf.add_n(tf.get_collection('losses'))
    trainstep=tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        print('im here')
        if os.path.exists('./modelvv_classify/checkpoint'):
            print('here load')
            saver.restore(sess, tf.train.latest_checkpoint('./modelvv_classify/'))

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        print('here,here')
        ops={'pointcloud_pl':pointcloud_pl,
             'label_pl':label_pl,
             'pred':tf.argmax(pred,1),
             'accuracy':accuracy,
             'loss':loss,
             'trainstep':trainstep,
             'zhengze':zhengze,
             'merged':merged,
             'epoch':0,
             'is_training_pl':is_training_pl
            }
        for i in range(EPOCH_ITER_TIME):
            ops['epoch']=i
            train_one_epoch(sess,ops,writer)
            if (i+1)%50==0:
                save_path = saver.save(sess, './modelvv_classify/model',global_step=i)
        #eval_one_epoch(sess,ops) 
if __name__=='__main__':
    train()
