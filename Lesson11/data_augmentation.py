import cv2
import numpy as np
import tensorflow as tf

class DataAugmentation:
    def __init__(self, config):
        self.config = config

        return

    def preprocess(self, image, label):
        tf_mean = tf.constant(self.config.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(self.config.std, dtype=tf.float32, shape=[1, 1, 3])

        image = tf.cast(image, tf.float32)

        image -= tf_mean
        image /= tf_std

        return image, label