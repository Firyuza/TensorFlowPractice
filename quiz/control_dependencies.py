import tensorflow as tf

def control_dependencies_wrong_example():
    op = tf.add(1, 2)
    temp = tf.multiply(2, 3)
    with tf.control_dependencies([op]):
        return temp

def control_dependencies_right_example():
    op = tf.add(1, 2)
    with tf.control_dependencies([op]):
        return tf.multiply(2, 3)

def run_quiz():
    temp = control_dependencies_right_example() # control_dependencies_wrong_example()

    tensorboard_writer = tf.summary.FileWriter('./logs', tf.get_default_graph())

    session = tf.Session(graph=tf.get_default_graph())
    output = session.run(temp)
    print(output)

    return

run_quiz()

def run_batch(data):

    graph = tf.Graph()
    with graph.as_default():
        v1_ph = tf.placeholder(shape=[None, DIM], dtype=tf.float32, name='vector1')
        v2_ph = tf.placeholder(shape=[None, DIM], dtype=tf.float32, name='vector2')

        sub = tf.subtract(v1_ph, v2_ph)
        square = tf.square(sub)
        sum = tf.reduce_sum(square, axis=1)

        distance = tf.sqrt(sum)

    tf.print("distance:", distance, output_stream=sys.stdout)
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