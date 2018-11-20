from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops


class simpleModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(simpleModel, self).__init__()

        input_shape = [28, 28, 1]
        data_format = 'channels_last'
        self.reshape = tf.keras.layers.Reshape(target_shape=input_shape, input_shape=(input_shape[0] * input_shape[1],))

        self.conv1 = tf.keras.layers.Conv2D(16, 5, padding="same", activation='relu')
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)

        self.conv2 = tf.keras.layers.Conv2D(32, 5, padding="same", activation='relu')
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)

        self.conv3 = tf.keras.layers.Conv2D(64, 5, padding="same", activation='relu')
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)

        self.conv4 = tf.keras.layers.Conv2D(64, 5, padding="same", activation='relu')
        self.batch4 = tf.keras.layers.BatchNormalization()
        self.pool4 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)

        self.flat = tf.keras.layers.Flatten()
        self.fc5 = tf.keras.layers.Dense(1024, activation='relu')
        self.batch5 = tf.keras.layers.BatchNormalization()

        self.fc6 = tf.keras.layers.Dense(num_classes)
        self.batch6 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.reshape(inputs)

        x = self.conv1(x)
        x = self.batch1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x, training=training)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.batch4(x, training=training)
        x = self.pool4(x)

        x = self.flat(x)
        x = self.fc5(x)
        x = self.batch5(x, training=training)

        x = self.fc6(x)
        x = self.batch6(x, training=training)
        # x = tf.layers.dropout(x, rate=0.3, training=training)
        return x

    def get_acc(self, target):
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(target, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

    def get_loss(self):
        return self.loss

    def loss_fn(self, images, target, training):
        self.logits = self(images, training)  # call call(self, inputs, training=None) function
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=target))
        return self.loss

    def grads_fn(self, images, target, training):  # do not return loss and acc if unnecessary
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(images, target, training)
        return tape.gradient(loss, self.variables)


def create_model1():
    data_format = 'channels_last'
    input_shape = [28, 28, 1]
    l = tf.keras.layers
    max_pool = l.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)
    # The model consists of a sequential chain of layers, so tf.keras.Sequential (a subclass of tf.keras.Model) makes for a compact description.
    return tf.keras.Sequential(
        [
            l.Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
            l.Conv2D(16, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
            l.BatchNormalization(),
            max_pool,

            l.Conv2D(32, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
            l.BatchNormalization(),
            max_pool,

            l.Conv2D(64, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
            l.BatchNormalization(),
            max_pool,

            l.Conv2D(64, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
            l.BatchNormalization(),
            max_pool,

            l.Flatten(),
            l.Dense(1024, activation=tf.nn.relu),
            l.BatchNormalization(),

            # # l.Dropout(0.4),
            l.Dense(10),
            l.BatchNormalization()
        ])


def create_model2():
    data_format = 'channels_last'
    input_shape = [28, 28, 1]

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Reshape(target_shape=input_shape, input_shape=(input_shape[0] * input_shape[1],)))

    model.add(tf.keras.layers.Conv2D(16, 5, padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format))

    model.add(tf.keras.layers.Conv2D(32, 5, padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format))

    model.add(tf.keras.layers.Conv2D(64, 5, padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format))

    model.add(tf.keras.layers.Conv2D(64, 5, padding="same", activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.BatchNormalization())

    return model


def infer(images, is_training, class_num, height, width):
    images = tf.reshape(images, [-1, height, width, 1])
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}
                        ):
        conv1 = slim.conv2d(images, 16, [5, 5], scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2 = slim.conv2d(pool1, 32, [5, 5], scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

        conv3 = slim.conv2d(pool2, 64, [5, 5], scope='conv3')
        pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
        conv4 = slim.conv2d(pool3, 64, [5, 5], scope='conv4')
        pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')

        flatten = slim.flatten(pool4)
        fc = slim.fully_connected(flatten, 1024, scope='fc1')
        logits = slim.fully_connected(fc, class_num, activation_fn=None, scope='logits')
    return logits


def modelNoEager(images, labels, lr, is_training, class_num, height, width, bs):
    logits = infer(images, is_training, class_num, height, width)

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    step = tf.get_variable("step", [], initializer=tf.constant_initializer(1.0), trainable=False)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr[0])
    train_op = slim.learning.create_train_op(loss_op, optimizer, global_step=step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss_op = control_flow_ops.with_dependencies([updates], loss_op)

    return train_op, step, accuracy, loss_op, logits

