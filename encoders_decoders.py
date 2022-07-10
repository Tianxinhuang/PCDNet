'''
Created on February 4, 2017

@author: optas

'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout

from tf_utils import expand_scope_by_name, replicate_parameter_for_all_layers
EPOCH_ITER_TIME=101
BATCH_ITER_TIME=5000
BASE_LEARNING_RATE=0.001
REGULARIZATION_RATE=0.0001
BATCH_SIZE=16
DECAY_STEP=1000*BATCH_SIZE
DECAY_RATE=0.7


def get_weight_variable(shape,stddev,name,regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)):
    #print(shape)
    weight = tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram(name+'/weights',weight)
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weight))
    return weight
def get_bias_variable(shape,value,name):
    bias=tf.Variable(tf.constant(value, shape=shape, name=name,dtype=tf.float32))
    tf.summary.histogram(name+'/bias',bias)
    return bias
def get_learning_rate(step):
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, step,DECAY_STEP / BATCH_SIZE, DECAY_RATE, staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate
def conv2d(scope,inputs,num_outchannels,kernel_size,stride=[1,1],padding='SAME',stddev=1e-3,activation_func=tf.nn.relu):
    with tf.variable_scope(scope):
        kernel_h,kernel_w=kernel_size
        num_inchannels=inputs.get_shape()[-1].value
        kernel_shape=[kernel_h,kernel_w,num_inchannels,num_outchannels]
        kernel=get_weight_variable(kernel_shape,stddev,'weights')
        stride_h,stride_w=stride
        outputs=tf.nn.conv2d(inputs,kernel,[1,stride_h,stride_w,1],padding=padding)
        bias = get_bias_variable([num_outchannels],0,'bias')
      #  print(outputs,bias)
        outputs=tf.nn.bias_add(outputs,bias)
        if activation_func!=None:
            outputs=activation_func(outputs)
    return outputs
def fully_connect(scope,inputs,num_outputs,stddev=1e-3,activation_func=tf.nn.relu):
    num_inputs = inputs.get_shape()[-1].value
    # print(inputs,num_inputs)
    with tf.variable_scope(scope):
        weights=get_weight_variable([num_inputs,num_outputs],stddev=stddev,name='weights')
        bias=get_bias_variable([num_outputs],0,'bias')
        result=tf.nn.bias_add(tf.matmul(inputs,weights),bias)
    if(activation_func is not None):
        result=activation_func(result)
    return result

def get_direction(auchor_pts,input_feat,mlp=[64,64],use_network=False,only_axis=False):
    axis_direction=tf.constant([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],dtype=tf.float32)#6*3
    axis_direction=tf.reshape(axis_direction,[1,1,1,6,3])
    pt_direction_num=4
    exclude_loss=0
    if use_network:
        region_num=auchor_pts.get_shape()[1].value
        auchor_num=auchor_pts.get_shape()[2].value
        words=tf.concat([auchor_pts,tf.tile(input_feat,[1,region_num,auchor_num,1])],axis=-1)
        for i,outchannel in enumerate(mlp):
            words=conv2d('get_direction%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=None)
            #words = batch_normalization(words,name= 'basic_norm%d'%i)
            words=tf.nn.relu(words)
        words=conv2d('direction_out',words,pt_direction_num*3,[1,1],padding='VALID',activation_func=None)
        pt_direction=tf.reshape(words,[-1,region_num,auchor_num,pt_direction_num,3])
        pt_direction=pt_direction/(1e-8+tf.sqrt(tf.reduce_sum(tf.square(pt_direction),axis=-1,keepdims=True)))
        cosine_mat=tf.reduce_sum(tf.expand_dims(pt_direction,axis=3)*tf.expand_dims(pt_direction,axis=4),axis=-1)
        sub_mat=tf.reshape(tf.eye(pt_direction_num),[1,1,1,pt_direction_num,pt_direction_num])
        sub_mat=tf.tile(sub_mat,[tf.shape(cosine_mat)[0],region_num,auchor_num,1,1])
        cosine_mat-=sub_mat
        print(cosine_mat)
        #dismat=tf.reduce_sum(tf.square(tf.expand_dims(pt_direction,axis=3)-tf.expand_dims(pt_direction,axis=4)),axis=-1)
        #print(cosine_mat)
        #exclude_loss=tf.reduce_max(1/(1+dismat),axis=[-1,-2])#batch*region_num*auchor_num
        exclude_loss=tf.reduce_max(cosine_mat,axis=[-1,-2])#batch*region_num*auchor_num
    else:
        pt_direction=tf.constant([[1,1,1],[1,1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[1,-1,1],[1,-1,-1]],dtype=tf.float32)#8*3
        pt_direction=pt_direction/tf.sqrt(tf.reduce_sum(tf.square(pt_direction),axis=-1,keepdims=True))#14*3
        pt_direction=tf.reshape(pt_direction,[1,1,1,-1,3])
    #directions=tf.concat([tf.tile(axis_direction,[tf.shape(pt_direction)[0],1,tf.shape(pt_direction)[2],1,1]),pt_direction],axis=-2)
    directions=pt_direction 
    #directions=tf.reshape(directions,[1,1,1,14,3])
    if only_axis:
        return axis_direction
    return directions,exclude_loss
def encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True,b_norm_decay=0.9, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''

    if verbose:
        print ('Building Encoder')

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    #if n_layers < 2:
    #    raise ValueError('More than 1 layers are expected.')

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        #if verbose:
        #    print (name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            #if i<n_layers-1:
            layer = batch_normalization(layer,decay=b_norm_decay, name=name, reuse=reuse, scope=scope_i)
        #    if verbose:
        #        print( 'bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print (layer)
        #    print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')
    layers=layer
    if symmetry is not None:
        layer = symmetry(layer, axis=1)
        if verbose:
            print (layer)

    if closing is not None:
        layer = closing(layer)
        print (layer)

    return layer


def decoder_with_fc_only(latent_signal, layer_sizes=[], b_norm=True,b_norm_decay=0.9,b_norm_decay_finish=0.9, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print('Building Decoder')

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')

    for i in range(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            layer = latent_signal

        layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print (name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer,decay=b_norm_decay, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print (layer)
            print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    if verbose:
        print (name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer,decay=b_norm_decay_finish, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

    if verbose:
        print (layer)
        print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    return layer
def decoder_with_folding_only(latent_signal, layer_sizes=[], b_norm=True,b_norm_decay=0.9,b_norm_decay_finish=0.9, non_linearity=tf.nn.relu,
                         regularizer=None, weight_decay=0.001, reuse=False, scope=None, dropout_prob=None,
                         b_norm_finish=False, verbose=False):
    '''A decoding network which maps points from the latent space back onto the data space.
    '''
    if verbose:
        print('Building Decoder')

    n_layers = len(layer_sizes)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('For an FC decoder with single a layer use simpler code.')
    
    for i in range(0, n_layers - 1):
        name = 'decoder_fc_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        if i == 0:
            grid_feat=latent_signal
            xlength,ylength=1.0,1.0
            xgrid_size,ygrid_size=45,45
            #xgrid_size,ygrid_size=8,8
            ptnum=int(xgrid_size*ygrid_size)
            xgrid_feat=-xlength+2*xlength*tf.tile(tf.reshape(tf.linspace(0.0,1.0,xgrid_size),[1,1,-1,1]),[tf.shape(grid_feat)[0],ygrid_size,1,1])#batch*1*xgrid*1
            ygrid_feat=-ylength+2*ylength*tf.tile(tf.reshape(tf.linspace(0.0,1.0,ygrid_size),[1,-1,1,1]),[tf.shape(grid_feat)[0],1,xgrid_size,1])#batch*1*ygrid*1
            #xgrid_feat=tf.tile(tf.reshape(xgrid_feat,[-1,xgrid_size,1,1]),[1,1,ygrid_size,1])
            #ygrid_feat=tf.tile(ygrid_feat,[1,xgrid_size,1,1])
            grid_feat=tf.concat([xgrid_feat,ygrid_feat],axis=-1)
            grid_feat=tf.reshape(grid_feat,[-1,ptnum,2])
            #grid_feat=tf.concat([tf.tile(xgrid_feat,[1,1,ygrid_size,1]),tf.reshape(tf.tile(ygrid_feat,[1,1,1,xgrid_size]),[-1,ptnum,up_ratio,1])],axis=-1)#batch*1*up_ratio*2
            #grid_feat=tf.squeeze(grid_feat,[1])
            layer = tf.concat([grid_feat,tf.tile(tf.expand_dims(latent_signal,axis=1),[1,grid_feat.get_shape()[1].value,1])],axis=-1)
        layer=conv_1d(layer, nb_filter=layer_sizes[i], filter_size=1, strides=1, regularizer=None,
                        weight_decay=0.001, name='fd_%d'%i, reuse=False, scope=scope_i, padding='same')
        #layer = fully_connected(layer, layer_sizes[i], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)

        if verbose:
            print (name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer,decay=b_norm_decay, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None:
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print (layer)
            print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    # Last decoding layer never has a non-linearity.
    name = 'decoder_fc_' + str(n_layers - 1)
    scope_i = expand_scope_by_name(scope, name)
    layer = conv_1d(layer, nb_filter=layer_sizes[-1], filter_size=1, strides=1, regularizer=None,
                        weight_decay=0.001, name='fd1_out', reuse=False, scope=scope_i, padding='same')
    layer=tf.concat([layer,tf.tile(tf.expand_dims(latent_signal,axis=1),[1,layer.get_shape()[1].value,1])],axis=-1)
    #layer = fully_connected(layer, layer_sizes[n_layers - 1], activation='linear', weights_init='xavier', name=name, regularizer=regularizer, weight_decay=weight_decay, reuse=reuse, scope=scope_i)
    for i, outchannel in enumerate([256,256]):
    #for i, outchannel in enumerate([128,128]):
        layer=conv_1d(layer, nb_filter=outchannel, filter_size=1, strides=1, regularizer=None,
                        weight_decay=0.001, name='fd2_%d'%i, reuse=False, scope=scope_i, padding='same')
    layer = conv_1d(layer, nb_filter=layer_sizes[-1], filter_size=1, strides=1, regularizer=None,
                        weight_decay=0.001, name='fd2_out', reuse=False, scope=scope_i, padding='same')
    if verbose:
        print (name, 'FC params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

    if b_norm_finish:
        name += '_bnorm'
        scope_i = expand_scope_by_name(scope, name)
        layer = batch_normalization(layer,decay=b_norm_decay_finish, name=name, reuse=reuse, scope=scope_i)
        if verbose:
            print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

    if verbose:
        print (layer)
        print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    return layer


def decoder_with_convs_only(in_signal, n_filters, filter_sizes, strides, padding='same', b_norm=True, non_linearity=tf.nn.relu,
                            conv_op=conv_1d, regularizer=None, weight_decay=0.001, dropout_prob=None, upsample_sizes=None,
                            b_norm_finish=False, scope=None, reuse=False, verbose=False):

    if verbose:
        print ('Building Decoder')

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    for i in range(n_layers):
        if i == 0:
            layer = in_signal

        name = 'decoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)

        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i],
                        strides=strides[i], padding=padding, regularizer=regularizer, weight_decay=weight_decay,
                        name=name, reuse=reuse, scope=scope_i)

        if verbose:
            print (name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()),)

        if (b_norm and i < n_layers - 1) or (i == n_layers - 1 and b_norm_finish):
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print ('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None and i < n_layers - 1:  # Last layer doesn't have a non-linearity.
            layer = non_linearity(layer)

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if upsample_sizes is not None and upsample_sizes[i] is not None:
            layer = tf.tile(layer, multiples=[1, upsample_sizes[i], 1])

        if verbose:
            print (layer)
            print ('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')

    return layer
