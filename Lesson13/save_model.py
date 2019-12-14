import tensorflow as tf
import tensorflow_addons as tfa
import os
import h5py


def save_variables(self, network, optimizer, model_dir, step):
    print('Saving variables')

    file_path = os.path.join(model_dir, 'model-%d.h5' % step)
    file = h5py.File(file_path, 'w')
    weight = network.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight' + str(i), data=weight[i])
    file.close()

    file_path = os.path.join(model_dir, 'optimizer-%d.h5' % step)
    file = h5py.File(file_path, 'w')
    weight = optimizer.get_weights()
    for i in range(len(weight)):
        file.create_dataset('weight' + str(i), data=weight[i])
    file.close()

    return

def load_variables(self):
    if cfg.train.restore_model_path != '':
        print('Loading variables')

        file = h5py.File(cfg.train.restore_model_path, 'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)].value)
        self.network.set_weights(weight)

        file = h5py.File(cfg.train.restore_model_path.replace('model-', 'optimizer-'), 'r')
        weight = []
        for i in range(1):#len(file.keys())):
            weight.append(file['weight' + str(i)].value)
        self.optimizer.set_weights(weight)

        step = int(cfg.train.restore_model_path.split('.h5')[0].split('-')[-1])
        self.optimizer.iterations.assign(step)
    else:
        step = 0

    return step