from __future__ import division, print_function, absolute_import
import os, sys
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tabular as tb
import itertools

from tfutils import base, data, model, optimizer, utils

from train_retina_all import ln, cnn

stim_type = 'naturalscene'
#stim_type = 'whitenoise'

if stim_type == 'whitenoise':
    TOTAL_BATCH_SIZE = 5957
    MB_SIZE = 5957
    NUM_GPUS = 1
else:
    TOTAL_BATCH_SIZE = 5956
    MB_SIZE = 5956
    NUM_GPUS = 1
    
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
print('NUM BATCHES PER EPOCH:', NUM_BATCHES_PER_EPOCH)
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
    
class RetinaDataExperiment():
    """
    Defines the neural data testing experiment
    """
    class Config():
        target_layers = ['conv1',
                         'conv2',
                         'fc']
        #extraction_step = 50
        exp_id = 'trainval0'
        #data_path = '/datasets/deepretina_data/tf_records/' + stim_type
        batch_size = 5957
        seed = 8 # Group number
        gfs_targets = [] 
        val_steps = N_TEST // MB_SIZE + 1,

    def __init__(self, extraction_step, model_stim_type, data_stim_type, model_type):
        self.extraction_step = extraction_step
        self.model_stim_type = model_stim_type
        self.data_stim_type = data_stim_type
        self.model_type = model_type
        self.test_data_path = '/datasets/deepretina_data/tf_records/' + data_stim_type

    def setup_params(self):
        params = {}

        params['validation_params'] = {
            'valid0': {
                'data_params': {
                    'func': retinaTF,
                    #'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
                    'source_dirs': [os.path.join(self.test_data_path, 'images'), os.path.join(self.test_data_path, 'labels')],
                    'resize': IMAGE_SIZE_RESIZE,
                    'data_path': self.test_data_path,
                    'file_pattern': 'test*.tfrecords', #'*.tfrecords', 
                    'batch_size': INPUT_BATCH_SIZE,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed, 
                    'n_threads': 4,
                },
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size':MB_SIZE,
                    'seed': self.Config.seed,
                    'capacity': 11*INPUT_BATCH_SIZE,
                    'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                },
                'targets': {
                    'func': self.pearsonAggFunc,
                    'target': 'labels',
                },
                'num_steps': N_TEST // MB_SIZE + 1,
                'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},  #self.retina_analysis,
                'online_agg_func': self.online_agg,
            }
        }

        params['model_params'] = {
            'func': cnn,
            #'func': ln,
            'num_gpus': NUM_GPUS,
            'devices': DEVICES,
            'prefix': MODEL_PREFIX
        }

        params['save_params'] = {
            'host': 'localhost',
            'port': 27017, #24444
            'dbname': self.model_type,
            'collname': self.model_stim_type + '_' + self.data_stim_type,
            'exp_id': self.Config.exp_id + '_' + str(self.extraction_step),
        }

        params['load_params'] = {
            'host': 'localhost',
            'port': 27017, #24444
            'dbname': self.model_type,
            'collname': self.model_stim_type,
            'exp_id': self.Config.exp_id,
            'do_restore': True,
            'query': {'step': self.extraction_step} \
                    if self.extraction_step is not None else None,
            'db_restore': False,
        }

        params['inter_op_parallelism_threads'] = 500

        return params

    def poisson_loss(self, logits, labels):
        epsilon = 1e-8
        logits = logits["pred"]
        #N = logits.get_shape().as_list()[1]
        #loss = 1.0/N * tf.reduce_sum(logits - labels * tf.log(logits + epsilon), 1)
        loss = logits - labels * tf.log(logits + epsilon)
        return loss
    
    def mean_loss_with_reg(self, loss):
        return tf.reduce_mean(loss) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    def loss_metric(self, inputs, outputs, target, **kwargs):
        metrics_dict = {}
        metrics_dict['poisson_loss'] = self.mean_loss_with_reg(self.poisson_loss(outputs, inputs[target]), **kwargs)
        return metrics_dict

    def pearson_correlation_py(self, logits, labels):
        s_logits = np.mean(logits, axis=0)
        s_labels = np.mean(labels, axis=0)
        std_logits = np.std(logits, axis=0)
        std_labels = np.std(labels, axis=0)
        r = np.mean((logits - s_logits)*(labels - s_labels), axis=0)/(std_logits * std_labels)        
        
        print("Pearson:")
        print(r)
        return r

    def pearson_correlation(self, logits, labels):
        print("Pearson Corr")
        return tf.py_func(self.pearson_correlation_py, [logits, labels], tf.float32)
        
    def pearsonAggFunc(self, inputs, outputs, **kwargs):
        print("returnOutputs")
        metrics_dict = {}
        key = self.data_stim_type + "_testcorr"
        metrics_dict[key] = self.pearson_correlation(logits=outputs['pred'], labels=inputs['labels'])
        return metrics_dict

    def online_agg(self, agg_res, res, step):
        """
        Appends the value for each key
        """
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            if 'kernel' in k:
                agg_res[k] = v
            else:
                agg_res[k].append(v)
        return agg_res


if __name__ == '__main__':
    base.get_params()
    #model_type = 'ln_model'
    model_type = 'cnn'
    #extraction_steps = np.linspace(0, 600, 13, dtype=int)
    extraction_steps = np.linspace(0, 1000, 21, dtype=int)
    model_stim_types = ['whitenoise', 'naturalscene']
    data_stim_types = ['whitenoise', 'naturalscene']
    for extraction_step in extraction_steps:
        for model_stim_type in model_stim_types:
            for data_stim_type in data_stim_types:
                print("GETTING DATA FOR: " + str(extraction_step) + '_' + model_stim_type + '_' + data_stim_type)
                m = RetinaDataExperiment(extraction_step, model_stim_type, data_stim_type, model_type)
                params = m.setup_params()
                base.test_from_params(**params)


