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