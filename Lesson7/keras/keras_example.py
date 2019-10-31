import numpy as np
import tensorflow as tf

from keras.config import cfg

class KerasExample:
    def __init__(self):
        self.__load_data()
        self.__create_graph()

        return

    def __load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.y_train = self.y_train.reshape([-1])
        self.y_test = self.y_test.reshape([-1])

        self.nrof_classes = cfg.dataset.nrof_classes
        return

    def __preperare_train_data(self, image, label):
        tf_mean = tf.constant(cfg.train.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(cfg.train.std, dtype=tf.float32, shape=[1, 1, 3])

        image = tf.cast(image, tf.float32)

        image -= tf_mean
        image /= tf_std

        return image, label

    def __preperare_valid_data(self, image, label):
        tf_mean = tf.constant(cfg.train.mu, dtype=tf.float32, shape=[1, 1, 3])
        tf_std = tf.constant(cfg.train.std, dtype=tf.float32, shape=[1, 1, 3])

        image = tf.cast(image, tf.float32)

        image -= tf_mean
        image /= tf_std

        return image, label

    def __create_graph(self):
        self.input_tensor = tf.keras.layers.Input(shape=(32, 32, 3))

        # create the base pre-trained model
        self.base_model = tf.keras.applications.resnet50.ResNet50(input_tensor=self.input_tensor,
                                                         weights='imagenet',
                                                         include_top=False)

        # add a global spatial average pooling layer
        x = self.base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        self.predictions = tf.keras.layers.Dense(self.nrof_classes, activation='softmax')(x)

        # this is the model we will train
        self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=self.predictions)

        if cfg.train.restore_model_path != '':
            self.model = self.__restore_model(cfg.train.restore_model_path)

        return

    def __restore_model(self, path):
        model = tf.keras.models.load_model(path)

        model.save(path)

        model.save_weights(path)

        print('Model restored')

        return model

    def __create_dataset(self, features, labels):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(map_func=self.__preperare_train_data,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=cfg.train.batch_size)
        dataset = dataset.shuffle(buffer_size=len(features)).prefetch(buffer_size=cfg.train.batch_size)

        return dataset

    def run_train(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir=cfg.train.logs_base_dir ),
            tf.keras.callbacks.ModelCheckpoint(cfg.train.models_base_dir + 'model_weights.{epoch:02d}-{val_loss:.2f}.h5',
                                               monitor='val_loss', verbose=0, save_best_only=False,
                                               save_weights_only=False, mode='auto', period=1)
        ]

        self.model.compile(optimizer=tf.keras.optimizers.SGD(lr=cfg.train.learning_rate, momentum=0.9),
                           loss=tf.keras.losses.sparse_categorical_crossentropy,
                           metrics=[tf.keras.metrics.sparse_categorical_accuracy])

        self.model.fit(self.__create_dataset(self.x_train, self.y_train),
                       epochs=cfg.train.nrof_epochs,
                       callbacks=callbacks,
                       steps_per_epoch=cfg.train.epoch_size,
                       validation_data=self.__create_dataset(self.x_test, self.y_test),
                       validation_steps=cfg.train.validation_steps)

        return

example = KerasExample()
example.run_train()