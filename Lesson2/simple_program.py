import sys
import tensorflow as tf
import numpy as np

DIM = 3
USE_GPU = True
GPU_MEMORY_FRACTION = 1.0

def run(data):
    graph = tf.Graph()
    with graph.as_default():
        v1_ph = tf.placeholder(shape=[DIM], dtype=tf.float32, name='vector1')
        v2_ph = tf.placeholder(shape=[DIM], dtype=tf.float32, name='vector2')

        sub = tf.subtract(v1_ph, v2_ph)
        square = tf.square(sub)
        sum = tf.reduce_sum(square)

        distance = tf.sqrt(sum)

        print_op = tf.print("distance:", distance, output_stream=sys.stdout)

        with tf.control_dependencies([print_op]):
            distance = distance * 0.

    if USE_GPU:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True,
                                      gpu_options=gpu_options)
    else:
        session_conf = tf.ConfigProto(
            device_count={'CPU': 1, 'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=True
        )
    session = tf.Session(graph=graph, config=session_conf)

    with session.as_default():
        distance_output = session.run(distance,
                                      feed_dict={v1_ph: data['v1'],
                                                 v2_ph: data['v2']})
        print(distance_output)

    return

def run_eager_execution():
    tf.enable_eager_execution()

    tensor = tf.range(10)
    tf.print("tensors:", tensor,  output_stream=sys.stdout)
    # tensors: [0 1 2 ... 7 8 9]

    print(np.square(tensor))
    # [ 0  1  4  9 16 25 36 49 64 81]

    tensor = tf.square(tensor)
    tf.print("tensors:", tensor, output_stream=sys.stdout)
    # tensors: [0 1 4 ... 49 64 81]

    tf.disable_eager_execution()

    return

run_eager_execution()

run({'v1': [1, 2, 3],
     'v2': [4, 5, 6]})

