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

        tf.summary.scalar('distance', distance)

        summary_merged_op = tf.summary.merge_all()
        tensorboard_writer = tf.summary.FileWriter('./logs', graph)

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
        summary_, distance_output, _ = session.run([summary_merged_op, distance, print_op],
                                      feed_dict={v1_ph: data['v1'],
                                                 v2_ph: data['v2']})
        tensorboard_writer.add_summary(summary_, 0)
        print(distance_output)

    return

run({'v1': [1, 2, 3],
     'v2': [4, 5, 6]})




