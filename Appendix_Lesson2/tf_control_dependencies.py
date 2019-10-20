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