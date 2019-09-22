import tensorflow as tf
import numpy as np

def graph_example():
    graph = tf.Graph()
    with graph.as_default():
        tmp = tf.constant(5.)

    assert tmp.graph is graph

    assert tmp.graph is not tf.get_default_graph()

    tmp = tf.constant(5.)
    assert tmp.graph is tf.get_default_graph()

    return

def placeholder_example():
    tmp = tf.placeholder(dtype=tf.int32, shape=(None), name='temp_variable')
    out = tf.matmul(tmp, tmp)

    session = tf.Session()
    output = session.run(out, feed_dict={tmp: np.ones((2, 2))})

    print(output)  # [[2 2] [2 2]]

    return

def variable_example():
    temp = tf.Variable(initial_value=5., dtype=tf.float32,
                       trainable=False, name='temp')

    output_op = tf.multiply(temp, temp)

    print(output_op) # Tensor("Mul:0", shape=(), dtype=float32)

    init_variables_op = tf.initialize_all_variables()

    session = tf.Session()

    session.run(init_variables_op)
    output_op = session.run(output_op)

    print(output_op) # 25.0

    return

def session_example():
    tmp = tf.multiply(2, 3)

    session = tf.Session()
    output = session.run(tmp)

    print(output)  # 6

    graph = tf.Graph()
    with graph.as_default():
        tmp = tf.multiply(2, 3)

    session = tf.Session(graph=graph)
    output = session.run(tmp)

    print(output)  # 6

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    session_conf = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False,
                                  gpu_options=gpu_options)

    session_conf = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )
    session = tf.Session(graph=graph, config=session_conf)

    output = session.run(tmp)

    print(output)  # 6

    return