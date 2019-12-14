import tensorflow as tf
import gc
import time
import os
import mlflow

from tensorboard.plugins.hparams import api as hp

from Lesson12.networks import Network
from Lesson12.config import cfg
from Lesson12.data_loader import DataLoader
from Lesson12.tf_logging import log_config
from Lesson12.custom_accuracy import CustomSparseCategoricalAccuracy
from Lesson12.validation import Validation

mlflow.set_experiment(cfg.train.experiment_name)

class ImageClassification:
    def __init__(self):

        self.data_loader = DataLoader(cfg)

        self.__build_model()

        self.validation = Validation(self.data_loader, self.valid_file_writer, cfg.validation)

        mlflow.log_param('lr', cfg.train.learning_rate)
        mlflow.log_param('optimizer', cfg.train.optimizer)

        return

    def __build_model(self):
        self.net = Network(cfg.train.weight_decay, cfg.dataset.nrof_classes)
        self.net.build((None, cfg.train.image_size,  cfg.train.image_size, cfg.train.image_channels))
        self.net.summary()

        lr_schedule = tf.optimizers.schedules.ExponentialDecay(
            cfg.train.learning_rate,
            decay_steps=cfg.train.learning_rate_decay_steps,
            decay_rate=cfg.train.learning_rate_decay_factor,
            staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.CE_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.category_metric = tf.metrics.SparseCategoricalAccuracy()
        self.custom_category_metric = CustomSparseCategoricalAccuracy('custom_category_metric')

        self.valid_file_writer = tf.summary.create_file_writer(cfg.train.logs_base_dir + "/valid")
        self.train_file_writer = tf.summary.create_file_writer(cfg.train.logs_base_dir + "/train")
        self.train_file_writer.set_as_default()

        with self.train_file_writer.as_default():
            hp.hparams({'image_size': cfg.train.image_size,
                        'optimizer': cfg.train.optimizer,
                        'weight_decay': cfg.train.weight_decay})

        return

    def __save_model(self, model_dir, step):
        # Save the model checkpoint
        print('Saving variables')
        start_time = time.time()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        checkpoint_path = os.path.join(model_dir, 'model-%d.h5' % step)

        self.net.save_weights(checkpoint_path)

        save_time_variables = time.time() - start_time
        print('Variables saved in %.2f seconds' % save_time_variables)

        return

    def __reset_all_metrics(self):
        self.category_metric.reset_states()
        self.custom_category_metric.reset_states()

        return

    def run_train_epoch(self, epoch):
        global_step = self.optimizer.iterations.numpy()
        progbar = tf.keras.utils.Progbar(self.data_loader.epoch_size)
        batch_id = 0
        for images, labels in self.data_loader.dataset:
            with tf.GradientTape() as tape:
                logits = self.net(images, True)
                self.prediction = tf.nn.softmax(logits)

                self.category_metric.update_state(labels, self.prediction)
                self.custom_category_metric.update_state(labels, self.prediction)

                ce_loss_value = self.CE_loss(labels, self.prediction)
                reg_loss = cfg.train.weight_decay * tf.add_n([tf.nn.l2_loss(w) for w in self.net.trainable_variables])
                total_loss_value = ce_loss_value + reg_loss

            grads = tape.gradient(total_loss_value, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))

            with self.train_file_writer.as_default():
                tf.summary.scalar('reg_loss', reg_loss, global_step)
                tf.summary.scalar('CE_loss', ce_loss_value.numpy(), global_step)
                tf.summary.scalar('total_loss', total_loss_value.numpy(), global_step)
                tf.summary.scalar('category_accuracy', self.category_metric.result().numpy(), global_step)
                tf.summary.scalar('learning_rate', self.optimizer.learning_rate.__call__(global_step).numpy(), global_step)

                tf.summary.image('training_images', images[:4], step=global_step)

            progbar.update(batch_id, [('total_loss', total_loss_value), ('ce_loss_value', ce_loss_value),
                                      ('category_accuracy', self.category_metric.result().numpy()),
                                      ('custom_category_accuracy', self.custom_category_metric.result().numpy())])

            global_step += 1
            batch_id += 1

        return global_step

    def run_train(self):
        if cfg.train.restore_model_path != '':
            self.net.load_weights(cfg.train.restore_model_path, by_name=True)
            self.net.summary()
            print('Model restored')

            try:
                step = int(cfg.train.restore_loss_model_path.split('/')[-1].split('-')[-1].split('.')[0])
            except:
                step = 0

            self.optimizer.iterations.assign(step)
            global_step = self.optimizer.iterations.numpy()
            print(global_step)
        else:
            step = 0
        gc.collect()

        log_config('config', cfg, self.train_file_writer, step)

        epoch = 0
        while epoch < cfg.train.epoch_size:
            print('EPOCH %d'% epoch)

            self.__reset_all_metrics()

            step = self.run_train_epoch(epoch)
            self.__save_model(cfg.train.models_base_dir + str(epoch) + '/', step)

            self.validation.run_validation(self.net, self.category_metric, step)

            epoch += 1
            gc.collect()

        return

with mlflow.start_run(run_name=cfg.train.run_name):
    mlflow.log_artifact(os.path.abspath(__file__))

    img_net = ImageClassification()
    img_net.run_train()