'''
Created on September 2, 2017

@author: optas
'''
import numpy as np
import tensorflow as tf
import random
from encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only,conv2d,fully_connect,batch_normalization,get_direction,decoder_with_folding_only
from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
def sampling(npoint,xyz,use_type='f'):
    if use_type=='f':
        bnum=tf.shape(xyz)[0]
        idx=tf_sampling.farthest_point_sample(npoint, xyz)
        #new_xyz=tf_sampling.gather_point(xyz,idx)
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    elif use_type=='r':
        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=np.arange(ptnum)
        ptids=tf.random_shuffle(ptids,seed=None)
        #random.shuffle(ptids)
        #print(ptids,ptnum,npoint)
        #ptidsc=ptids[tf.py_func(np.random.choice(ptnum,npoint,replace=False),tf.int32)]

        #ptidsc=ptids[:npoint]
        ptidsc=tf.gather(ptids,tf.cast(tf.range(npoint),tf.int32))
        ptid=tf.cast(tf.tile(tf.reshape(ptidsc,[-1,npoint,1]),[bnum,1,1]),tf.int32)
        #ptid=tf.tile(tf.constant(ptidsc,shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])

        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    elif use_type=='n':
        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=np.arange(ptnum)
        ptidsc=tf.gather(ptids,tf.cast(tf.range(npoint),tf.int32))
        ptid=tf.cast(tf.tile(tf.reshape(ptidsc,[-1,npoint,1]),[bnum,1,1]),tf.int32)
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    return idx,new_xyz
def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, dnum, bneck_post_mlp=False,mode='fc'):
    ''' Single class experiments.
    '''
    #if n_pc_points != 2048:
    #    raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    #decoder = decoder_with_fc_only
    #decoder = decoder_with_folding_only

    n_input = [n_pc_points, dnum]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True,
                    'non_linearity':tf.nn.relu
                    }
    if mode=='fc':
        decoder = decoder_with_fc_only
        decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': True
                        }
    else:
        decoder = decoder_with_folding_only
        decoder_args = {'layer_sizes': [256,256,3],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': True
                        }
    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args
 
def movenet(inpts,knum=64,mlp1=[128,128],mlp2=[128,128],startcen=None,infer=False):
    #with tf.variable_scope(scope):
    ptnum=inpts.get_shape()[1].value
    #_,startcen=sampling(knum,inpts,use_type='r')
    #_,allcen=sampling(ptnum,inpts,use_type='f')
    #_,startcen=sampling(knum,allcen,use_type='n')
    if startcen is None:
        if infer:
            startcen=sampling(knum,inpts,use_type='f')[1]
        else:
            _,allcen=sampling(ptnum,inpts,use_type='f')
            _,startcen=sampling(knum,allcen,use_type='n')
    words=tf.expand_dims(startcen,axis=2)
    inwords=tf.expand_dims(inpts,axis=2)

    for i,outchannel in enumerate(mlp1):
        inwords=conv2d('movein_state%d'%i,inwords,outchannel,[1,1],padding='VALID',activation_func=None)
        inwords=tf.nn.relu(inwords)

    inwords=tf.reduce_mean(inwords,axis=1,keepdims=True)#/ptnum

    for i,outchannel in enumerate(mlp1):
        words=conv2d('mover_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
        words=tf.nn.relu(words)

    wordsfeat=words
    words=tf.reduce_sum(words,axis=1,keepdims=True)/ptnum
    #words=tf.reduce_mean(words,axis=1,keepdims=True)

    words=tf.concat([tf.expand_dims(startcen,axis=2),tf.tile(words,[1,tf.shape(startcen)[1],1,1]),tf.tile(inwords,[1,tf.shape(startcen)[1],1,1])],axis=-1)
    for i,outchannel in enumerate(mlp2):
        words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
        words=tf.nn.relu(words)
    words=conv2d('basic_stateoutg',words,3,[1,1],padding='VALID',activation_func=None)
    move=tf.squeeze(words,[2])
    result=startcen+move
    movelen=tf.sqrt(tf.reduce_sum(tf.square(move),axis=-1))
    return result,movelen 
