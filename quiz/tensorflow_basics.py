import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    op1 = tf.multiply(2, 3)

with tf.Graph().as_default() as g2:
    op2 = tf.subtract(3, 1)

nrof_operations = len(tf.get_default_graph().get_operations())

print(nrof_operations)

