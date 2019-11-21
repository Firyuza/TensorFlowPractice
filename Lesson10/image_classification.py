import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import gc

from Lesson10.networks import Network
from Lesson10.config import cfg
from Lesson10.data_loader import DataLoader

class ImageClassification:
    def __init__(self):

        self.data_loader = DataLoader(cfg)

        self.__build_model()

        return

    def __build_model(self):
        self.net = None

        lr_schedule = tf.optimizers.schedules.ExponentialDecay(
            cfg.train.learning_rate,
            decay_steps=cfg.train.learning_rate_decay_steps,
            decay_rate=cfg.train.learning_rate_decay_factor,
            staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.CE_loss = None
        self.category_metric = None

        self.valid_file_writer = tf.summary.create_file_writer(cfg.train.logs_base_dir + "/valid")
        self.train_file_writer = tf.summary.create_file_writer(cfg.train.logs_base_dir + "/train")
        self.train_file_writer.set_as_default()

        return

    def __save_model(self, model_dir, step):
        # Save the model checkpoint
        print('Saving variables')
        start_time = time.time()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        checkpoint_path = os.path.join(model_dir, 'model-%d.h5' % step)

        # TODO save weights

        save_time_variables = time.time() - start_time
        print('Variables saved in %.2f seconds' % save_time_variables)

        return

    def run_train_epoch(self, epoch):
        global_step = self.optimizer.iterations.numpy()
        progbar = tf.keras.utils.Progbar(self.data_loader.epoch_size)

        batch_id = 0

        # TODO
        #     progbar.update(batch_id, [('total_loss', total_loss_value), ('ce_loss_value', ce_loss_value),
        #                                  ('category_accuracy', self.category_metric.result().numpy())])
        #
        #     global_step += 1
        #     batch_id += 1


        return global_step

    def run_train(self):
        if cfg.train.restore_model_path != '':
            # TODO
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

        epoch = 0
        while epoch < cfg.train.epoch_size:
            print('EPOCH %d'% epoch)
            step = self.run_train_epoch(epoch)
            self.__save_model(cfg.train.models_base_dir + str(epoch) + '/', step)

            # if (epoch+1) % 2 == 0:
            #     try:
            #         # TODO run validation
            #     except:
            #         print('Validation failed!')

            epoch += 1
            gc.collect()

        return

img_net = ImageClassification()
img_net.run_train()

