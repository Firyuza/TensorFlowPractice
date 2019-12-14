import numpy as np
import tensorflow as tf
import os

from mlflow import log_metric, log_artifact

from Lesson12.tf_logging import log_images

class Validation:
    def __init__(self, data_loader, valid_file_writer, cfg):
        self.data_loader = data_loader
        self.valid_file_writer = valid_file_writer
        self.cfg = cfg

        self.CE_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                               reduction=tf.losses.Reduction.SUM)

        log_artifact(os.path.abspath(__file__))

        return

    def run_validation(self, net, category_metric, step):
        print('Start validation')
        category_metric.reset_states()

        ce_loss_value = 0
        nrof_elements = 0
        progbar = tf.keras.utils.Progbar(self.data_loader.validation_epoch_size)
        batch_id = 0
        images_log = []
        labels_log = []
        predicted_labels_log = []
        for images, labels in self.data_loader.valid_dataset:
            logits = net(images, True)
            predictions = tf.nn.softmax(logits)

            category_metric.update_state(labels, predictions)

            ce_loss_value += self.CE_loss(labels, predictions)

            nrof_elements += len(labels)

            progbar.update(batch_id, [('ce_loss_value', ce_loss_value / nrof_elements),
                                      ('category_accuracy', category_metric.result().numpy())])

            batch_id += 1

            predicted_labels = tf.argmax(predictions, axis=1)
            incorrect_predictions = np.where(tf.squeeze(labels, 1).numpy() != predicted_labels)[0]

            images_log.extend(tf.gather(images, incorrect_predictions).numpy())
            labels_log.extend(tf.gather(tf.squeeze(labels, 1), incorrect_predictions).numpy())
            predicted_labels_log.extend(tf.gather(predicted_labels, incorrect_predictions).numpy())

        with self.valid_file_writer.as_default():
            tf.summary.scalar('CE_loss', ce_loss_value.numpy() / nrof_elements, step)
            tf.summary.scalar('category_accuracy', category_metric.result().numpy(), step)
            tf.summary.image("valid_images", log_images(images_log[:25], labels_log[:25], predicted_labels_log[:25],
                                                        self.data_loader.class_names), step)
            log_metric('validation_category_accuracy', category_metric.result().numpy(), step)

        return