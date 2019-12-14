import tensorflow as tf
import re
import itertools
# import tfplot
import numpy as np
import io

from textwrap import wrap
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def create_table(name, columns, rows, summary_writer, step):
    hyperparameters = [tf.convert_to_tensor(columns)]
    hyperparameters.extend([tf.convert_to_tensor([str(el) for el in row]) for row in rows])
    with summary_writer.as_default():
        tf.summary.text(name, tf.stack(hyperparameters), step=step)

    return

def log_config(name, config, summary_writer, step):
    general_keys = list(config.keys())
    rows = []
    for key in general_keys:
        try:
            subkeys = list(config[key])
            for subkey in subkeys:
                rows.append(['%s.%s' %(key, subkey), str(config[key][subkey])])
        except:
            rows.append([key, str(config[key])])

    hyperparameters = [tf.convert_to_tensor(row) for row in rows]
    with summary_writer.as_default():
        tf.summary.text(name, tf.stack(hyperparameters), step=step)

    return

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image

def image_grid(images, class_labels, predicted_class_labels, class_name_map):
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title='%s_%s' %
                                       (class_name_map[class_labels[i]],
                                        class_name_map[predicted_class_labels[i]]))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

    return figure

def log_images(images, class_labels, predicted_class_labels, class_names):
    figure = image_grid(images, class_labels, predicted_class_labels, class_names)

    return plot_to_image(figure)

def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)

    ''' confusion matrix summaries '''
    img_d_summary_dir = os.path.join(checkpoint_dir, "summaries", "img")
    img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir, sess.graph)
    img_d_summary = plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='dev/cm')
    img_d_summary_writer.add_summary(img_d_summary, current_step)

    return summary

