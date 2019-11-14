import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape(watch_accessed_variables=False) as g:
    g.watch(x)
    y = x * x
dy_dx = g.gradient(y, x)  # 6
print(0)



