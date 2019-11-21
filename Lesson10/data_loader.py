import tensorflow as tf

from .data_augmentation import DataAugmentation

class DataLoader:
    def __init__(self, cfg):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

        self.config = cfg

        self.data_augmentation = DataAugmentation(cfg.train)

        self._build_data_input_pipeline()

    def _build_data_input_pipeline(self):
        self.dataset = None

        return