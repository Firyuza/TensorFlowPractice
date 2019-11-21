import cv2
import numpy as np
import tensorflow as tf

class DataAugmentation:
    def __init__(self, config):
        self.config = config

        return

    def preprocess(self, image, label):
        # TODO

        return image, label