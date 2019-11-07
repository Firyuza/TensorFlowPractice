import tensorflow as tf
import os

tf.enable_eager_execution()

def gpu_example():
    a = tf.constant([1, 2, 3, 4, 5])
    b = tf.gather(a, [0, 1, 6])

    print(b.numpy())  # [1 2 0]

    return

def cpu_example():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    a = tf.constant([1, 2, 3, 4, 5])
    b = tf.gather(a, [0, 1, 6])

    print(b)  # throw Exception: InvalidArgumentError: indices[2] = 6 is not in [0, 5) [Op:GatherV2]

    return



