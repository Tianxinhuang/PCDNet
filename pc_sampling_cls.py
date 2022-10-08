import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
#import tf_util
import copy
import random

from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from ae_sam import mlp_architecture_ala_iclr_18,movenet
from tf_ops.grouping import tf_grouping
#query_ball_point, group_point
from provider import shuffle_data,shuffle_points,rotate_point_cloud,jitter_point_cloud
#from pointnet_cls import get_model,get_loss
from pointnet_cls import get_model,get_loss#,get_bn_decay
from samplenet import SoftProjection,get_project_loss
import argparse

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

#def train(ptnum=2048,tem=1.0,diswei=0.5):
def train(args):
    dnum=3
    pointcloud_pl=tf.placeholder(tf.float32,[args.batch_size,args.ptnum,dnum],name='pointcloud_pl')
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
        pred, end_points = get_model(inpts, is_training_pl, bn_decay=None)
        loss1 = get_loss(pred, label_pl, end_points)

    with tf.variable_scope('ge_1'):
        pred2, end_points = get_model(pointcloud_pl, is_training_pl, bn_decay=None)
        loss0 = get_loss(pred2, label_pl, end_points)
    lossd = entropy_distillation(pred, pred2, args.tem)

    loss_e=loss1+args.diswei*lossd+args.movewei*tf.reduce_mean(movelen)+args.simwei*get_aesimplify_loss(inpts,pointcloud_pl)+pjloss#+loss0#+0.1*KLs
    trainvars=tf.GraphKeys.GLOBAL_VARIABLES
        
    var1=tf.get_collection(trainvars,scope='sam')
    regularizer=tf.contrib.layers.l2_regularizer(0.0001)
    gezhengze=tf.reduce_sum([regularizer(v) for v in var1])
    loss_e=loss_e+args.reg*gezhengze
    trainstep=[]

    trainstep.append(tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss_e, global_step=global_step,var_list=var1))
    loss=[loss_e,loss1]

    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        gevar=tf.get_collection(trainvars,scope='ge')

        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        if os.path.exists(os.path.join(args.prepath,'checkpoint')):
            print('here load')
            for i in range(2):
                restore_into_scope(tf.train.latest_checkpoint(args.prepath), 'ge_'+str(i), sess)
            print('load completed')

        errlist=[]
        datalist=[]
        labelist=[]
        kklist=[np.power(2,v) for v in list(range(args.reso_start,args.reso_end))]
        kknum=len(kklist)
        #assert False
        plist=np.ones(kknum)/kknum
        eperr=[]
        dypara=args.interval
        for i in range(len(kklist)):
            errlist.append([])
            eperr.append([])

        trainfiles=getdata.getfile(os.path.join(args.filepath,'train_files.txt'))
        filenum=len(trainfiles)

        for j in range(filenum):
            data,label=getdata.load_h5label(os.path.join(args.filepath, trainfiles[j]))
            datalist.append(data)
            labelist.append(label)
        import copy
        for i in range(args.itertime):
            if i>0 and i%dypara==0:
                for k in range(kknum):
                    errlist[k].append(np.mean(eperr[k]))
                    eperr[k]=[]
            if i>0 and i%(2*dypara)==0:
                for k in range(kknum):
                    errs=errlist[k]
                    err=(max(errs)-min(errs))/max(errs)
                    plist[k]=np.exp(err)
                    plist=plist/sum(plist)
                for ii in range(len(kklist)):
                    errlist[ii]=[]
            for j in range(filenum):
                traindata,label=copy.deepcopy(datalist[j]),copy.deepcopy(labelist[j])
                traindata,label,_=shuffle_data(traindata[:,:args.ptnum],label)
                
                allnum=int(len(traindata)/args.batch_size)*args.batch_size
                batch_num=int(allnum/args.batch_size)
                
                for batch in range(batch_num):
                    start_idx = (batch * args.batch_size) % allnum
                    end_idx=(batch*args.batch_size)%allnum+args.batch_size
                    batch_point=traindata[start_idx:end_idx]
                    batch_point=shuffle_points(batch_point)
                    idx=np.random.choice(len(kklist),1,p=list(plist))[0]
                    kk=kklist[idx]

                    feed_dict = {pointcloud_pl: batch_point,knum:kk,is_training_pl:False,label_pl:np.squeeze(label[start_idx:end_idx])}
                    resi = sess.run([trainstep[0],loss], feed_dict=feed_dict)
                    losse=resi[1]

                    if (i+1)%dypara==0:
                        for kn in range(kknum):
                            eperr[kn].append(sess.run(loss1, feed_dict={pointcloud_pl: batch_point,knum:kklist[kn],is_training_pl:False,label_pl:np.squeeze(label[start_idx:end_idx])}))
                    if (batch+1) % args.seestep == 0:
                        print('sample num', kk)
                        print('epoch: %d '%i,'file: %d '%j,'batch: %d' %batch)
                        print('loss: ',resi[1])
                                                
            if (i+1)%args.savestep==0:
                save_path = tf.train.Saver(var_list=var1).save(sess, os.path.join(args.savepath,'model'),global_step=i)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptnum', type=int, default=2048, help='The number of points')
    parser.add_argument('--tem', type=float, default=1.0, help='The temperature of distillation')
    parser.add_argument('--diswei', type=float, default=0.5, help='The weights of distillation')
    parser.add_argument('--movewei', type=float, default=0.001, help='The weights of moving distances')
    parser.add_argument('--simwei', type=float, default=1.0, help='The weights of simiplify loss')
    parser.add_argument('--reso_start', type=int, default=5, help='Controlling the lowest resolution')
    parser.add_argument('--reso_end', type=int, default=12, help='Controlling the highest resolution')
    parser.add_argument('--interval', type=int, default=10, help='The interval of Dynamic resolution selection')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--savestep', type=int, default=100, help='The interval to save checkpoint')
    parser.add_argument('--seestep', type=int, default=16, help='The batch interval to see training errors')
    parser.add_argument('--itertime', type=int, default=1000, help='The number of epochs for iteration')
    parser.add_argument('--filepath', type=str, default='./data', help='The path of h5 training data')
    parser.add_argument('--savepath', type=str, default='./modelvv_samcls/', help='The path of saved checkpoint')
    parser.add_argument('--prepath', type=str, default='./pn_cls/', help='The checkpoint path of the pretrained cls network')
    parser.add_argument('--reg', type=float, default=0.001, help='The regularization parameter')
    parser.add_argument('--lr', type=float, default=0.0005, help='The learning rate')
    args=parser.parse_args()
    train(args)
