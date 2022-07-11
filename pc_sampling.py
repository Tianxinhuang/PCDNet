import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
import tf_util
import copy
import random
DATA_DIR=getdata.getspdir()
filelist=os.listdir(DATA_DIR)

from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from ae_sam import mlp_architecture_ala_iclr_18,movenet
from tf_ops.grouping import tf_grouping
from provider import shuffle_data,shuffle_points,rotate_point_cloud,jitter_point_cloud
from samplenet import SoftProjection,get_project_loss 
trainfiles=getdata.getfile(os.path.join(DATA_DIR,'train_files.txt'))

EPOCH_ITER_TIME=1000
BATCH_ITER_TIME=5000
BASE_LEARNING_RATE=0.01
REGULARIZATION_RATE=0.0001
BATCH_SIZE=16
DECAY_STEP=1000*BATCH_SIZE
DECAY_RATE=0.7
PT_NUM=2048
FILE_NUM=6
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.set_random_seed(1)
def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    #dist=tf.reduce_mean(tf.maximum(tf.sqrt(dist1),tf.sqrt(dist2)))
    dist=(dist1 + dist2)/2
    return dist
def chamfer_distilla(pcdi,pcdos,pcdot):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcdi, pcdos)
    dists1 = tf.sqrt(dist1)
    dists2 = tf.sqrt(dist2)

    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcdi, pcdot)
    distt1 = tf.sqrt(dist1)
    distt2 = tf.sqrt(dist2)

    dist1=tf.where(tf.greater(dists1+0.001,distt1),dists1,tf.zeros_like(dists1))
    dist2=tf.where(tf.greater(dists2+0.001,distt2),dists2,tf.zeros_like(dists2))

    dist=(tf.reduce_mean(dist1)+tf.reduce_mean(dist2))/2
    return dist

def sampling(npoint,xyz,use_type='f'):
    if use_type=='f':
        idx=tf_sampling.farthest_point_sample(npoint, xyz)
        new_xyz=tf_sampling.gather_point(xyz,idx)
    elif use_type=='r':
        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=arange(ptnum)
        random.shuffle(ptids)
        ptid=tf.tile(tf.constant(ptids[:npoint],shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    return new_xyz
def get_aesimplify_loss(sampts, data):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(sampts,data)
    maxdist=tf.reduce_mean(tf.reduce_max(dist1,axis=1))
    meandist=tf.reduce_mean(dist1,axis=[0,1])
    w=tf.cast(tf.shape(sampts)[1],tf.float32)
    loss=meandist+maxdist+w*tf.reduce_mean(dist2)/64
    #loss=meandist+maxdist+2*w*tf.reduce_mean(dist2)
    return loss
def get_clssimplify_loss(sampts, data):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(sampts,data)
    maxdist=tf.reduce_mean(tf.reduce_max(dist1,axis=1))
    meandist=tf.reduce_mean(dist1,axis=[0,1])
    w=sampts.get_shape()[1].value/30.0+0.5
    loss=meandist+maxdist+w*tf.reduce_mean(dist2)
    #loss=meandist+maxdist+2*w*tf.reduce_mean(dist2)
    return loss
def train():
    n_pc_points=2048
    ptnum=n_pc_points
    bneck_size=512
    featlen=64
    mlp=[64]
    mlp.append(2*featlen)
    mlp2=[128,128]
    cen_num=16
    region_num=1
    gregion=1
    rnum=1
    dnum=3
    pointcloud_pl=tf.placeholder(tf.float32,[None,PT_NUM,dnum],name='pointcloud_pl')
    knum=tf.placeholder(tf.float32,name='pointcloud_pl')
    global_step=tf.Variable(0,trainable=False)
    with tf.variable_scope('sam'):
        inpts,movelen=movenet(pointcloud_pl,knum=knum,mlp1=[128,256,256],mlp2=[128,128],startcen=None,infer=False)
        groupsize=16
        pj=SoftProjection(groupsize)
        proinpts,_,_=pj(pointcloud_pl,inpts,hard=False)
        pjloss=0.00001*get_project_loss(pj)

    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, 128,dnum,mode='fc')
    #with tf.variable_scope('ge'):
    #    word=encoder(inpts,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],strides=enc_args['strides'],b_norm=enc_args['b_norm'],b_norm_decay=1.0,verbose=enc_args['verbose'])
    #    out=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_decay=1.0,b_norm_finish=dec_args['b_norm_finish'],b_norm_decay_finish=1.0,verbose=dec_args['verbose'] )
    with tf.variable_scope('ge'):
        word=encoder(proinpts,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],strides=enc_args['strides'],b_norm=enc_args['b_norm'],b_norm_decay=1.0,verbose=enc_args['verbose'])
        outpj=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_decay=1.0,b_norm_finish=dec_args['b_norm_finish'],b_norm_decay_finish=1.0,verbose=dec_args['verbose'])
    with tf.variable_scope('ge',reuse=True):
        word=encoder(pointcloud_pl,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],strides=enc_args['strides'],b_norm=enc_args['b_norm'],b_norm_decay=1.0,verbose=enc_args['verbose'])
        outr=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_decay=1.0,b_norm_finish=dec_args['b_norm_finish'],b_norm_decay_finish=1.0,verbose=dec_args['verbose'])
    #out=tf.reshape(out,[-1,45*45,dnum])
    #out=tf.reshape(out,[-1,ptnum,dnum])
    outr=tf.reshape(outr,[-1,ptnum,dnum])
    outpj=tf.reshape(outpj,[-1,ptnum,dnum])
    zcons=0.001*tf.reduce_mean(tf.reduce_mean(tf.abs(word),axis=-1))

    #loss1=chamfer_big(pointcloud_pl,out)
    loss2=chamfer_big(pointcloud_pl,outpj)
    #loss0=chamfer_distilla(pointcloud_pl,outpj,outr)
    loss0=tf.where(tf.greater(loss2+0.001,chamfer_big(pointcloud_pl,outr)),loss2,tf.zeros_like(loss2))
    loss_e=loss2+loss0+0.001*tf.reduce_mean(movelen)+0.0001*get_aesimplify_loss(proinpts,pointcloud_pl)+pjloss
    #loss_e=loss2+loss0+0.001*get_aesimplify_loss(proinpts,pointcloud_pl)+pjloss
    trainvars=tf.GraphKeys.GLOBAL_VARIABLES

    var1=tf.get_collection(trainvars,scope='sam')
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    gezhengze=tf.reduce_sum([regularizer(v) for v in var1])
    loss_e=loss_e+0.001*gezhengze#//////////////////
    alldatanum=2048*FILE_NUM
    trainstep=[]
    trainstep.append(tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_e, global_step=global_step,var_list=var1))
    loss=[loss_e,pjloss]

    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.086
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        gevar=tf.get_collection(trainvars,scope='ge')
        #ge_saver=tf.train.Saver(var_list=gevar)
        ivar=[v for v in tf.get_collection(trainvars) if v.name.split(':')[0]=='is_training']

        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        #tf.train.Saver(var_list=var3+var5).restore(sess, tf.train.latest_checkpoint('./best_lnfc/'))
        if os.path.exists('./fc_cd/checkpoint'):
            print('here load')
            tf.train.Saver(var_list=gevar).restore(sess, tf.train.latest_checkpoint('./fc_cd'))

        errlist=[]
        datalist=[]
        kklist=[np.power(2,v) for v in list(range(5,12))]
        kknum=len(kklist)
        plist=np.ones(kknum)/kknum
        eperr=[]
        dypara=10
        for i in range(len(kklist)):
            errlist.append([])
            eperr.append([])
        for j in range(FILE_NUM):
            datalist.append(getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j])))
        for i in range(EPOCH_ITER_TIME):
            if i>0 and i%dypara==0:
                for k in range(kknum):
                    errlist[k].append(np.mean(eperr[k]))
                    eperr[k]=[]
            if i>0 and i%(2*dypara)==0:
                for k in range(kknum):
                    errs=errlist[k]
                    err=(max(errs)-min(errs))/max(errs)
                    plist[k]=np.exp(err)
                    print(err)
                plist=plist/sum(plist)
                for ii in range(len(kklist)):
                    errlist[ii]=[]
                print(plist)
            for j in range(FILE_NUM):
                traindata = datalist[j]                                                
                ids=list(range(len(traindata)))
                random.shuffle(ids)
                traindata=traindata[ids,:,:]
                traindata=shuffle_points(traindata[:,:PT_NUM])
                
                allnum=int(len(traindata)/BATCH_SIZE)*BATCH_SIZE
                batch_num=int(allnum/BATCH_SIZE)
                
                for batch in range(batch_num):
                    start_idx = (batch * BATCH_SIZE) % allnum
                    end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
                    batch_point=traindata[start_idx:end_idx]
                    kklist=[np.power(2,v) for v in list(range(5,12))]
                    idx=np.random.choice(len(kklist),1,p=list(plist))[0]
                    #print(idx)
                    kk=kklist[idx]
                    #print(kk)
                    feed_dict = {pointcloud_pl: batch_point,knum:kk}
                    resi = sess.run([trainstep[0],loss], feed_dict=feed_dict)
                    losse=resi[1]
                    if (i+1)%dypara==0:
                        for kn in range(kknum):
                            eperr[kn].append(sess.run(loss2, feed_dict={pointcloud_pl: batch_point,knum:kklist[kn]}))
                    errlist[idx].append(losse[-1])

                    if (batch+1) % 16 == 0:
                        print('sample num', kk)
                        print('epoch: %d '%i,'file: %d '%j,'batch: %d' %batch)
                        print('loss: ',resi[1])
                                                
            if (i+1)%100==0:
                save_path = saver.save(sess, './modelvv_sam/model',global_step=i)
if __name__=='__main__':
    train()
