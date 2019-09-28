import tensorflow as tf

def control_dependencies_example():
    multiply_op = tf.multiply(2, 3)
    subtract_op = tf.subtract(multiply_op, 3)

    with tf.control_dependencies([multiply_op, subtract_op]):
        # Operations constructed here will be executed after
        # multiply_op, subtract_op
        divide_op = tf.divide(subtract_op, multiply_op)

    output_op = tf.add(divide_op, divide_op)

    session = tf.Session()
    output = session.run(output_op)

    print(output) # 1.0

    return


def control_dependencies_None_example():
    multiply_op = tf.multiply(2, 3)
    subtract_op = tf.subtract(multiply_op, 3)

    with tf.control_dependencies([multiply_op, subtract_op]):
        # Operations constructed here will be executed after
        # multiply_op, subtract_op
        divide_op = tf.divide(subtract_op, multiply_op)
        with tf.control_dependencies(None):
            # Operations constructed here will not not wait
            # ANY operation
            output_op = tf.add(divide_op, divide_op)

    session = tf.Session()
    output = session.run(output_op)

    print(output)  # 1.0

    return