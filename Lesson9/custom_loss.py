import tensorflow as tf

class MaxMargin(tf.keras.losses.Loss):
    def __init__(self, delta, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='max_margin'):
        super(MaxMargin, self).__init__(reduction=reduction,
                                        name=name)
        self.delta = delta

    def call(self, v1, v2):
        norm = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(v1, v2, axis=1), axis=1), axis=1))

        return tf.maximum(0., self.delta - norm)