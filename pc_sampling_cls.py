import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
#import tf_util
import copy
import random
import point_choose
DATA_DIR=getdata.getdir()
filelist=os.listdir(DATA_DIR)

from tf_ops2.emd import tf_auctionmatch
from tf_ops2.CD import tf_nndistance
from tf_ops2.sampling import tf_sampling
from ae_sam import mlp_architecture_ala_iclr_18,adaptive_loss_net,local_loss_net,local_loss_net2,cen_net,reverse_net,find_diff,FC_layer,mlp_architecture_sam,errnet,movenet
from tf_ops2.grouping import tf_grouping
#query_ball_point, group_point
from provider import shuffle_data,shuffle_points,rotate_point_cloud,jitter_point_cloud
#from pointnet_cls import get_model,get_loss
from pointnet_cls import get_model,get_loss#,get_bn_decay
from samplenet import SoftProjection,get_project_loss

trainfiles=getdata.getfile(os.path.join(DATA_DIR,'train_files.txt'))
#testfiles=getdata.getfile(os.path.join(DATA_DIR,'test_files.txt'))

EPOCH_ITER_TIME=2000
BATCH_ITER_TIME=5000
BASE_LEARNING_RATE=0.01
REGULARIZATION_RATE=0.0001
BATCH_SIZE=16
DECAY_STEP=1000*BATCH_SIZE
DECAY_RATE=0.7
PT_NUM=2048
FILE_NUM=5
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.set_random_seed(1)
def restore_into_scope(model_path, scope_name, sess):
    # restored_vars = get_tensors_in_checkpoint_file(file_name=MODEL_PATH)
    # tensors_to_load = build_tensors_in_checkpoint_file(restored_vars, scope=scope_name)

    global_vars = tf.global_variables()
    tensors_to_load = [v for v in global_vars if v.name.startswith(scope_name + "/")]

    load_dict = {}
    for j in range(0, np.size(tensors_to_load)):
        tensor_name = tensors_to_load[j].name
        #print(tensor_name)
        tensor_name = tensor_name[0:-2]  # remove ':0'
        tensor_name = tensor_name.replace(scope_name + "/", "ge/")  # remove scope
        #print(tensor_name)
        load_dict.update({tensor_name: tensors_to_load[j]})
    #print(load_dict)
    loader = tf.train.Saver(var_list=load_dict)
    loader.restore(sess, model_path)
    print(
        "Model restored from: {0} into scope: {1}.".format(model_path, scope_name)
    )
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
def entropy_distillation(pred1,pred2,t):
    result=tf.reduce_mean(tf.reduce_sum(-tf.nn.softmax(pred1/t)*tf.log(tf.nn.softmax(pred2/t)+1e-5),axis=-1))*(t**2)
    return result
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
    pointcloud_pl=tf.placeholder(tf.float32,[BATCH_SIZE,PT_NUM,dnum],name='pointcloud_pl')
    label_pl=tf.placeholder(tf.int32,[None],name='label_pl')
    is_training_pl = tf.placeholder(tf.bool, shape=())
    knum=tf.placeholder(tf.float32,name='pointcloud_pl')
    global_step=tf.Variable(0,trainable=False)
    with tf.variable_scope('sam'):
        inpts0,movelen=movenet(pointcloud_pl,knum=knum,mlp1=[128,256,256],mlp2=[128,128])
        groupsize=16
        pj=SoftProjection(groupsize)
        inpts,_,_=pj(pointcloud_pl,inpts0,hard=False)
        pjloss=1.0*get_project_loss(pj)

    with tf.variable_scope('ge_0'):
        #inpts=tf.reshape(inpts,[BATCH_SIZE,-1,3])
        pred, end_points = get_model(inpts, is_training_pl, bn_decay=None)
        loss1 = get_loss(pred, label_pl, end_points)

    with tf.variable_scope('ge_1'):
        #inpts=tf.reshape(inpts,[BATCH_SIZE,-1,3])
        pred2, end_points = get_model(pointcloud_pl, is_training_pl, bn_decay=None)
        loss0 = get_loss(pred2, label_pl, end_points)
    lossd = entropy_distillation(pred, pred2,1.0)

    loss_e=loss1+0.5*lossd+0.001*tf.reduce_mean(movelen)+get_aesimplify_loss(inpts,pointcloud_pl)+pjloss#+loss0#+0.1*KLs
    trainvars=tf.GraphKeys.GLOBAL_VARIABLES
        
    var1=tf.get_collection(trainvars,scope='sam')
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    gezhengze=tf.reduce_sum([regularizer(v) for v in var1])
    loss_e=loss_e+0.001*gezhengze#//////////////////
    alldatanum=2048*FILE_NUM
    trainstep=[]
    #lr=tf.train.exponential_decay(0.01, global_step, (EPOCH_ITER_TIME*alldatanum)/(BATCH_SIZE*10), 0.96, staircase=True)
    trainstep.append(tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss_e, global_step=global_step,var_list=var1))
    loss=[loss_e,loss1]

    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.086
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        gevar=tf.get_collection(trainvars,scope='ge')
        #ge_saver=tf.train.Saver(var_list=gevar)


        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        #tf.train.Saver(var_list=var3+var5).restore(sess, tf.train.latest_checkpoint('./best_lnfc/'))
        if os.path.exists('./pn_cls/checkpoint'):
            print('here load')
            #tf.train.Saver(var_list=gevar).restore(sess, tf.train.latest_checkpoint('./modelvv_classify'))
            for i in range(2):
                restore_into_scope(tf.train.latest_checkpoint('./pn_cls'), 'ge_'+str(i), sess)
            print('load completed')
        #from tflearn import is_training
        #is_training(True, session=sess)
        #if os.path.exists('./CD_0.0001/checkpoint'):
        #    print('here load')
        #    tf.train.Saver(var_list=gevar).restore(sess, tf.train.latest_checkpoint('./CD_0.0001'))
            #saver.restore(sess, tf.train.latest_checkpoint('./modelvv_sam/'))

        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter("logs/", sess.graph)
        #klist=[]
        sig=0
        sig2=0
        lastval=0
        lastlocal=0
        #oastrl=0
        losse,lossd,lossd2=0,0,0
        loss_global,loss_local,localzhengze,meanloss,reverloss,loss_local1,loss_local2=0,0,0,0,0,0,0
        cyclenum=1
        rlin=100
        reverloss=100
        lastrl=0
        import time
        errlist=[]
        datalist=[]
        labelist=[]
        kklist=[np.power(2,v) for v in list(range(5,12))]
        kknum=len(kklist)
        #assert False
        plist=np.ones(kknum)/kknum
        eperr=[]
        dypara=10
        for i in range(len(kklist)):
            errlist.append([])
            eperr.append([])
        for j in range(FILE_NUM):
            #datalist.append(getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j])))
            data,label=getdata.load_h5label(os.path.join(DATA_DIR, trainfiles[j]))
            datalist.append(data)
            labelist.append(label)
        #tlist=[]
        for i in range(EPOCH_ITER_TIME):
            if i>0 and i%dypara==0:
                for k in range(kknum):
                    errlist[k].append(np.mean(eperr[k]))
                    eperr[k]=[]
            if i>0 and i%(2*dypara)==0:
                for k in range(kknum):
                    errs=errlist[k]
                    err=(max(errs)-min(errs))#/max(errs)
                    plist[k]=np.exp(err)
                    print(err)
                plist=plist/sum(plist)
                for ii in range(len(kklist)):
                    errlist[ii]=[]
                print(plist)
            tlist=[]
            for j in range(FILE_NUM):
                #traindata = getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j]))
                #traindata,label = getdata.load_h5label(os.path.join(DATA_DIR, trainfiles[j]))
                traindata,label=datalist[j],labelist[j]
                traindata,label,_=shuffle_data(traindata[:,:PT_NUM],label)
                #colors = getdata.load_color(os.path.join(DATA_DIR, trainfiles[j]))
                #print(colors)
                #assert False
                #traindata=concatenate([traindata,colors],axis=-1)
                                
                #random.shuffle(traindata)
                #traindata.swapaxes(1,0)
                #random.shuffle(traindata)
                #traindata.swapaxes(1,0)

                allnum=int(len(traindata)/BATCH_SIZE)*BATCH_SIZE
                batch_num=int(allnum/BATCH_SIZE)
                
                for batch in range(batch_num):
                    start_idx = (batch * BATCH_SIZE) % allnum
                    end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
                    #start_idx=0
                    #end_idx=BATCH_SIZE
                    batch_point=traindata[start_idx:end_idx]
                    batch_point=shuffle_points(batch_point)
                    #random.shuffle(batch_point)
                    for ei in range(cyclenum):
                        #kk=int(1024*np.random.rand(1)+64)
                        #kk=np.power(2,int(5+6*np.random.rand(1)))
                        idx=np.random.choice(len(kklist),1,p=list(plist))[0]
                        #print(idx)
                        kk=kklist[idx]

                        feed_dict = {pointcloud_pl: batch_point,knum:kk,is_training_pl:False,label_pl:np.squeeze(label[start_idx:end_idx])}
                        stime=time.time()
                        resi = sess.run([trainstep[0],loss], feed_dict=feed_dict)
                        etime=time.time()
                        tlist.append(etime-stime)
                        losse=resi[1]

                        if (i+1)%dypara==0:
                            for kn in range(kknum):
                                eperr[kn].append(sess.run(loss1, feed_dict={pointcloud_pl: batch_point,knum:kklist[kn],is_training_pl:False,label_pl:np.squeeze(label[start_idx:end_idx])}))
                    if (batch+1) % 16 == 0:
                        print('sample num', kk)
                        print('mean time', mean(tlist))
                        print('epoch: %d '%i,'file: %d '%j,'batch: %d' %batch)
                        print('loss: ',resi[1])
                        #assert False
            #print('mean time', mean(tlist))
                        
            if (i+1)%100==0:
                print('mean time', mean(tlist))
                save_path = tf.train.Saver(var_list=var1).save(sess, './modelvv_samcls2/model',global_step=i)
if __name__=='__main__':
    train()
