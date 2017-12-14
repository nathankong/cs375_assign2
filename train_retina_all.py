from __future__ import division, print_function, absolute_import
import os, sys
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
import copy

## THINGS TO CHANGE!
# toggle this to train or to validate at the end
train_net = True
# toggle this to train on whitenoise or naturalscene data
#stim_type = 'whitenoise'
stim_type = 'naturalscene'

# Figure out the hostname
host = os.uname()[1]
if 'instance-1' in host:
    if train_net:
        print('In train mode...')
        TOTAL_BATCH_SIZE = 5000
        MB_SIZE = 5000
        NUM_GPUS = 1
    else:
        print('In val mode...')
        if stim_type == 'whitenoise':
            TOTAL_BATCH_SIZE = 5957
            MB_SIZE = 5957
            NUM_GPUS = 1
        else:
            TOTAL_BATCH_SIZE = 5956
            MB_SIZE = 5956
            NUM_GPUS = 1
            
else:
    print("Data path not found!!")
    exit()

if not isinstance(NUM_GPUS, list):
    DEVICES = ['/gpu:' + str(i) for i in range(NUM_GPUS)]
else:
    DEVICES = ['/gpu:' + str(i) for i in range(len(NUM_GPUS))]

MODEL_PREFIX = 'model_0'

# Data parameters
if stim_type == 'whitenoise':
    N_TRAIN = 323762
    N_TEST = 5957
else:
    N_TRAIN = 323756
    N_TEST = 5956

INPUT_BATCH_SIZE = 1024 # queue size
OUTPUT_BATCH_SIZE = TOTAL_BATCH_SIZE
print('TOTAL BATCH SIZE:', OUTPUT_BATCH_SIZE)
NUM_BATCHES_PER_EPOCH = N_TRAIN // OUTPUT_BATCH_SIZE
IMAGE_SIZE_RESIZE = 50

DATA_PATH = '/datasets/deepretina_data/tf_records/' + stim_type
print('Data path: ', DATA_PATH)

# data provider
class retinaTF(data.TFRecordsParallelByFileProvider):
    def __init__(self,
                 source_dirs,
                 resize=IMAGE_SIZE_RESIZE,
                 **kwargs):
        if resize is None:
            self.resize = 50
        else:
            self.resize = resize

        postprocess = {'images': [], 'labels': []}
        postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['images'].insert(1, (tf.reshape, ([-1] + [50, 50, 40], ), {}))
        postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 
    
        postprocess['labels'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['labels'].insert(1, (tf.reshape, ([-1] + [5], ), {}))

        super(retinaTF, self).__init__(
              source_dirs,
              postprocess=postprocess,
              **kwargs)


    def postproc_imgs(self, ims):
        def _postprocess_images(im):
            im = tf.image.resize_images(im, [self.resize, self.resize])
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

def ln(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, seed=0, cfg_final=None):
    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]
    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size
    
    X = tf.reshape(inputs['images'], [batch_size, -1], 'flat_input')
    n_in = X.get_shape().as_list()[-1]
    n_rgc = 5
    out = {}
    with tf.variable_scope('linear'):
        W = tf.get_variable('weights', [n_in, n_rgc], tf.float32, 
                            tf.contrib.layers.xavier_initializer(), 
                            tf.contrib.layers.l2_regularizer(1e-3))
        b = tf.get_variable('bias', [n_rgc], tf.float32,
                             tf.constant_initializer())
        h = tf.nn.bias_add(tf.matmul(X, W), b, name='linear')
        out['weights'] = W
    with tf.variable_scope('non_linear'):
        y = tf.nn.softplus(h, 'non_linear')
        out['pred'] = y

    return out, params

def cnn(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, seed=0, cfg_final=None):
    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]
    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size
    outputs = inputs

    # implement your CNN here
    with tf.variable_scope("conv1"):
        inputData = inputs['images']
        filtersShape = [15,15,40,16]
        biasShape = [16]
        strideShape = [1,1,1,1]
        weights = tf.get_variable('W1', filtersShape, tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3))
        bias = tf.get_variable('b1', biasShape, tf.float32, tf.constant_initializer(0))
        conv = tf.nn.conv2d(inputData, weights, strides=strideShape, padding="VALID") 
        noise = tf.random_normal(shape=tf.shape(conv), mean=0.0, stddev=0.1,dtype=tf.float32) 
        if train:
            outputs["conv1"] = tf.nn.relu(conv + noise + bias)
        else:
            outputs["conv1"] = tf.nn.relu(conv + bias)
        outputs["conv1_kernel"] = weights
    
    print("CONV1 OUTPUT SIZE: ", outputs["conv1"].get_shape().as_list())
    with tf.variable_scope("conv2"):
        inputData = outputs["conv1"]
        filtersShape = [9,9,16,8]
        biasShape = [8]
        strideShape = [1,1,1,1]
        weights = tf.get_variable('W2', filtersShape, tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3))
        bias = tf.get_variable('b2', biasShape, tf.float32, tf.constant_initializer(0))
        conv = tf.nn.conv2d(inputData, weights, strides=strideShape, padding="VALID") 
        noise = tf.random_normal(shape=tf.shape(conv), mean=0.0, stddev=0.1,dtype=tf.float32) 
        if train:
            outputs["conv2"] = tf.nn.relu(conv + noise + bias)
        else:
            outputs["conv2"] = tf.nn.relu(conv + bias)
        outputs["conv2_kernel"] = weights

    print("CONV2 OUTPUT SIZE: ", outputs["conv2"].get_shape().as_list())
    with tf.variable_scope("fc"):
        inputData = outputs["conv2"]
        fc_length = int(inputData.shape[1] * inputData.shape[2] * inputData.shape[3])
        fc_flatten = tf.reshape(inputData, (int(inputData.shape[0]), fc_length))


        filtersShape = [fc_length, 5]
        biasShape = [5]
        weights = tf.get_variable('W3', filtersShape, tf.float32,
                        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.05),
                        regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3))
        bias = tf.get_variable('b3', biasShape, tf.float32, tf.constant_initializer(0))
        outputs["fc"] = tf.nn.softplus(tf.nn.bias_add(tf.matmul(fc_flatten, weights), bias))
        outputs["pred"] = outputs["fc"]

    sys.stdout.flush()
    return outputs, params

def poisson_loss(logits, labels):
    epsilon = 1e-8
    logits = logits["pred"]
    #N = logits.get_shape().as_list()[1]
    #loss = 1.0/N * tf.reduce_sum(logits - labels * tf.log(logits + epsilon), 1)
    loss = logits - labels * tf.log(logits + epsilon)
    return loss
    
def mean_loss_with_reg(loss):
    return tf.reduce_mean(loss) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def loss_metric(inputs, outputs, target, **kwargs):
    metrics_dict = {}
    metrics_dict['poisson_loss'] = mean_loss_with_reg(poisson_loss(logits=outputs, labels=inputs[target]), **kwargs)
    return metrics_dict

def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    print('KEYS: ', keys)
    for k in keys:
        plucked = [d[k] for d in step_results]
        if isinstance(k, str) and 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval

def pearson_correlation_py(logits, labels):
    s_logits = np.mean(logits, axis=0)
    s_labels = np.mean(labels, axis=0)
    std_logits = np.std(logits, axis=0)
    std_labels = np.std(labels, axis=0)
    r = np.sum((logits - s_logits)*(labels - s_labels), axis=0)/(std_logits * std_labels)
    return r

def pearson_correlation(logits, labels):
    return tf.py_func(pearson_correlation_py, [logits, labels], tf.float32)

# model parameters

default_params = {
    'save_params': {
        'host': 'localhost',
        'port': 27017,
        'dbname': 'deepretina',
        'collname': stim_type,
        'exp_id': 'trainval0',

        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 50,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 50,
        'save_filters_freq': 50,
        'cache_filters_freq': 50,
        # 'cache_dir': None,  # defaults to '~/.tfutils'
    },

    'load_params': {
        'do_restore': True,
        'query': None
    },

    'model_params': {
        'func': ln,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    },

    'train_params': {
        'minibatch_size': MB_SIZE,
        'data_params': {
            'func': retinaTF,
            'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
            'resize': IMAGE_SIZE_RESIZE,
            'batch_size': INPUT_BATCH_SIZE,
            'file_pattern': 'train*.tfrecords',
            'n_threads': 4
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': OUTPUT_BATCH_SIZE,
            'capacity': 11*INPUT_BATCH_SIZE,
            'min_after_dequeue': 10*INPUT_BATCH_SIZE,
            'seed': 0,
        },
        'thres_loss': float('inf'),
        'num_steps': 50 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'validate_first': True,
    },

    'loss_params': {
        'targets': ['labels'],
        'agg_func': mean_loss_with_reg,
        'loss_per_case_func': poisson_loss
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 1e-3,
        'decay_rate': 1.0, # constant learning rate
        'decay_steps': NUM_BATCHES_PER_EPOCH,
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.AdamOptimizer,
        'clip': True,
        'trainable_names': None
    },

    'validation_params': {
        'test_loss': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_metric,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': N_TEST // MB_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        },
        'train_loss': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'train*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_metric,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': N_TRAIN // OUTPUT_BATCH_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        }

    },
    'log_device_placement': False,  # if variable placement has to be logged
}

def train_ln():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'ln_model'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    params['model_params'] = {
        'func': ln,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    }

    # 1e-4 for natural scenes
    #params['learning_rate_params']['learning_rate'] = 1e-3
    params['learning_rate_params']['learning_rate'] = 1e-4
    base.train_from_params(**params)

def train_cnn():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'cnn'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    params['model_params'] = {
        'func': cnn,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    }

    # 1e-4 for natural scenes
    #params['learning_rate_params']['learning_rate'] = 1e-3
    params['learning_rate_params']['learning_rate'] = 1e-4
    base.train_from_params(**params)
 
if __name__ == '__main__':
    train_cnn()
    #train_ln()


