import tensorflow as tf

from Lesson11.data_augmentation import DataAugmentation

class DataLoader:
    def __init__(self, cfg):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

        self.config = cfg

        self.data_augmentation = DataAugmentation(cfg.train)

        self._build_data_input_pipeline()

        self.epoch_size = len(self.x_train) // cfg.train.batch_size

    def _build_data_input_pipeline(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.shuffle(buffer_size=len(self.x_train))
        dataset = dataset.map(map_func=self.data_augmentation.preprocess,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = dataset.batch(batch_size=self.config.train.batch_size).prefetch(buffer_size=self.config.train.prefetch_buffer_size)

        return