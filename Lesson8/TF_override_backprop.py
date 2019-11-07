import tensorflow as tf
import numpy as np

def custom_add(c, k, d, graph, name=None):
    with tf.name_scope(name, "AddGrad", [c, k, d]) as name:
        # Need to generate a unique name to avoid duplicates
        # if you have more than one custom gradients:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e+8))

        inputs = [c, k, d]

        tf.RegisterGradient(rnd_name)(backprop_func)
        with graph.gradient_override_map({"PyFunc": rnd_name}):
            return tf.py_func(forward_func, inputs, [np.float32], name=name)

def forward_func(c, k, d):
    e = np.add(c, k)

    return e.astype(np.float32)

def backprop_func(op, grad):
    c = op.inputs[0]
    k = op.inputs[1]
    d = op.inputs[2]
    e = tf.add(c, k)
    return (e + d) * grad , d * grad, e * grad


stddev = 1e-1
labels = [0, 1]
data = [1, 2]

labels_ph = tf.placeholder(shape=(2), dtype=tf.int32)
data_ph = tf.placeholder(shape=(2), dtype=tf.float32)

c = tf.Variable(tf.truncated_normal(shape=[2], stddev=stddev))
k = tf.Variable(tf.truncated_normal(shape=[2], stddev=stddev))

d = tf.add(data_ph, c)
e = custom_add(c, k, d, tf.get_default_graph()) # tf.add(c, k)
a = tf.multiply(d, e)

loss = tf.losses.softmax_cross_entropy(logits=a, onehot_labels=labels_ph)
opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=0.1)

grads = opt.compute_gradients(loss, tf.trainable_variables())
grads = list(grads)
train_op = opt.apply_gradients(grads_and_vars=grads)

session = tf.Session()
session.run(tf.global_variables_initializer())

session.run(train_op, feed_dict={labels_ph: labels,
                                 data_ph: data})

