import tensorflow as tf

print('Wrong example')
x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    y = x
init = tf.initialize_all_variables()

with tf.Session() as session:
    init.run()
    for _ in range(5):
        print(session.run(y))

tf.reset_default_graph()

print('Right example')
x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)

with tf.control_dependencies([x_plus_1]):
    y = tf.identity(x) # x_plus_1
init = tf.initialize_all_variables()

with tf.Session() as session:
    init.run()
    for _ in range(5):
        print(session.run(y))