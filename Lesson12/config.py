import numpy as np
from easydict import EasyDict

cfg = EasyDict()

cfg.dataset = EasyDict()

cfg.dataset.nrof_classes = 10

cfg.train = EasyDict()

cfg.train.experiment_name = 'ImageClassification'
cfg.train.run_name = 'First Train'

cfg.train.nrof_classes = 10

cfg.train.image_size = 32
cfg.train.image_channels = 3
cfg.train.batch_size = 32
cfg.train.epoch_size = 5000

cfg.train.prefetch_buffer_size = cfg.train.batch_size

cfg.train.mu = np.asarray([[[125.306918046875, 122.950394140625, 113.86538318359375]]])
cfg.train.std = np.asarray([[[62.993219278136934, 62.08870764001416, 66.70489964063094]]])

cfg.train.use_gpu = True
cfg.train.gpu_memory_fraction = 1.0

cfg.train.optimizer = 'ADAM'
cfg.train.learning_rate = 3e-5
cfg.train.learning_rate_decay_steps = 10000
cfg.train.learning_rate_decay_factor = 0.98

cfg.train.weight_decay = 0.0005

cfg.train.augmentation_probability = 0.4

cfg.train.restore_model_path = ''

cfg.train.logs_base_dir = './logs'
cfg.train.models_base_dir = './models/'

cfg.validation = EasyDict()

cfg.validation.batch_size = 128

def iterate_over_config(dictionary, last_key=None):
    for key, value in dictionary.items():
        new_key_name = key if last_key == None else last_key + '.' + key

        if isinstance(value, dict): iterate_over_config(value, new_key_name)
        else:
            if not isinstance(value, str):
                value = value.__str__()
            yield [new_key_name, value]

