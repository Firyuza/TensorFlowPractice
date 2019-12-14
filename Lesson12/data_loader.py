import tensorflow as tf

from Lesson12.data_augmentation import DataAugmentation

class DataLoader:
    def __init__(self, cfg):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

        self.config = cfg

        self.class_names = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'}

        self.data_augmentation = DataAugmentation(cfg.train)

        self._build_train_data_input_pipeline()

        self._build_valid_data_input_pipeline()

        self.epoch_size = len(self.x_train) // cfg.train.batch_size

        self.validation_epoch_size = len(self.x_test) // cfg.validation.batch_size

    def _build_train_data_input_pipeline(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.shuffle(buffer_size=len(self.x_train))
        dataset = dataset.map(map_func=self.data_augmentation.preprocess,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = dataset.batch(batch_size=self.config.train.batch_size)\
            .prefetch(buffer_size=self.config.train.prefetch_buffer_size)

        return

    def _build_valid_data_input_pipeline(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        dataset = dataset.map(map_func=self.data_augmentation.preprocess,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.valid_dataset = dataset.batch(batch_size=self.config.validation.batch_size)\
            .prefetch(buffer_size=self.config.train.prefetch_buffer_size)

        return