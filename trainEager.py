from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math


from tensorflow.examples.tutorials.mnist import input_data
from model import *
from saverestore import *

import tensorflow as tf
import tensorflow.contrib.eager as tfe

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def trainEagerV2(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    model = simpleModel(num_classes=10)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr[0])
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)

    total_loss, total_acc = 0.0, 0.0
    for i in range(1, args.epoch + 1):
        imgs, labs = mnist.train.next_batch(args.batch_size)

        grads = model.grads_fn(imgs, labs, True)
        optimizer.apply_gradients(zip(grads, model.variables), step_counter)

        loss_value = model.get_loss()
        acc = model.get_acc(labs)

        total_loss += loss_value
        total_acc += acc

        if i % args.save_step ==0:
            str = 'train: {:.0f}\t{:.4f} {:.4f}\t{:.4f} {:.4f}'.format(i, total_loss / i, total_acc / i, loss_value, acc)
            print(str)
            evaluateEagerV2(args, mnist, model, i)
            saveModelV1(args.resultFolder, model, step_counter)
            #saveModelV2(args.resultFolder, checkpoint)


def testEagerV2(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    model = simpleModel(num_classes=10)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr[0])

    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)

    restoreModelV1(args.resultFolder, model)
    # restoreModelV2(args.resultFolder, checkpoint)

    evaluateEagerV2(args, mnist, model, 0)


def evaluateEagerV2(args, dataset, model, epoch):
    loss, acc = 0.0, 0.0
    num_images = len(dataset.test.images)
    nrof_batches = math.ceil(num_images / args.batch_size)
    for i in range(nrof_batches):
        imgs, labs = dataset.test.next_batch(args.batch_size)

        # logits = model(imgs, training=False)  # call model.call(self, inputs, training=None) function
        _loss = model.loss_fn(imgs, labs, training=False)
        _acc = model.get_acc(labs)

        loss += _loss * args.batch_size
        acc += _acc * args.batch_size

    str = 'evaluate: {} {:.4f} {:.4f}'.format(epoch, loss / num_images, acc / num_images)
    print(str)


def trainEagerV1(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    model = create_model1()
    # model = create_model2()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr[0])

    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)

    total_loss, total_acc = 0.0, 0.0
    for i in range(1, args.epoch + 1):
        imgs, labs = mnist.train.next_batch(args.batch_size)

        with tf.GradientTape() as tape:
            logits = model(imgs, training=True)
            loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labs))
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labs, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        total_loss += loss_value
        total_acc += acc

        if i % args.save_step ==0:
            str = 'train: {:.0f}\t{:.4f} {:.4f}\t{:.4f} {:.4f}'.format(i, total_loss / i, total_acc / i, loss_value, acc)
            print(str)
            evaluateEagerV1(args, mnist, model, i)
            # saveModelV1(args.resultFolder, model, step_counter)
            saveModelV2(args.resultFolder, checkpoint)


def testEagerV1(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    # model = create_model1()
    model = create_model2()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr[0])

    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)

    # restoreModelV1(args.resultFolder, model)
    restoreModelV2(args.resultFolder, checkpoint)

    evaluateEagerV1(args, mnist, model, 0)


def evaluateEagerV1(args, dataset, model, epoch):
    loss, acc = 0.0, 0.0
    num_images = len(dataset.test.images)
    nrof_batches = math.ceil(num_images / args.batch_size)
    for i in range(nrof_batches):
        imgs, labs = dataset.test.next_batch(args.batch_size)

        logits = model(imgs, training=False)
        _loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labs))
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labs, 1))
        _acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        loss += _loss * args.batch_size
        acc += _acc * args.batch_size

    str = 'evaluate: {} {:.4f} {:.4f}'.format(epoch, loss / num_images, acc / num_images)
    print(str)


def trainNoEager(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    image_placeholder = tf.placeholder(tf.float32, [None, 784])
    label_placeholder = tf.placeholder(tf.float32, [None, 10])
    lr_placeholder = tf.placeholder(tf.float32, [None], name='learning_rate')
    bs_placeholder = tf.placeholder(tf.int64, name='batch_size')

    is_training = tf.placeholder(tf.bool, [])

    train_op, global_step, accuracy, loss_op, logits = modelNoEager(image_placeholder, label_placeholder,
        lr_placeholder, is_training, args.classNum, args.height, args.width, bs_placeholder)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=args.saveMaxModelNum)
    sess.run(tf.global_variables_initializer())

    total_loss, total_acc = 0.0, 0.0
    for i in range(args.epoch + 1):
        imgs, labs = mnist.train.next_batch(args.batch_size)
        train_dict = {image_placeholder: imgs, label_placeholder: labs, is_training: True, lr_placeholder:args.lr, bs_placeholder:args.batch_size}
        step, _, entropy, acc, _logit = sess.run([global_step, train_op, loss_op, accuracy, logits], feed_dict=train_dict)

        total_loss += entropy
        total_acc += acc

        if i % args.save_step ==0:
            str = '{:.0f}\t{:.4f} {:.4f}\t{:.4f} {:.4f}'.format(step, total_loss / step, total_acc / step, entropy, acc)
            print(str)
            evaluateNoEager(args, sess, mnist, step, image_placeholder, label_placeholder, is_training, loss_op, accuracy, bs_placeholder)
            saveModelNoEager(sess, saver, args.resultFolder, 'model', int(step))

    sess.close()


def testNoEager(args):
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    image_placeholder = tf.placeholder(tf.float32, [None, 784])
    label_placeholder = tf.placeholder(tf.float32, [None, 10])
    lr_placeholder = tf.placeholder(tf.float32, [None], name='learning_rate')
    bs_placeholder = tf.placeholder(tf.int64, name='batch_size')

    is_training = tf.placeholder(tf.bool, [])

    _, _, accuracy, loss_op, logits = modelNoEager(image_placeholder, label_placeholder, lr_placeholder,
        is_training, args.classNum, args.height, args.width, bs_placeholder)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=args.saveMaxModelNum)
    sess.run(tf.global_variables_initializer())

    restoreModelNoEager(sess, saver, args.resultFolder)

    evaluateNoEager(args, sess, mnist, 0, image_placeholder, label_placeholder, is_training, loss_op, accuracy, bs_placeholder)

    sess.close()


def evaluateNoEager(args, sess, dataset, epoch, image_placeholder, label_placeholder, is_training, loss_op, accuracy_op, bs_placeholder):
    loss, acc = 0.0, 0.0
    num_images = len(dataset.test.images)
    nrof_batches = math.ceil(num_images / args.batch_size)
    for i in range(nrof_batches):
        imgs, labs = dataset.test.next_batch(args.batch_size)

        feed_dict = {image_placeholder: imgs, label_placeholder: labs, is_training: False, bs_placeholder:args.batch_size}
        _loss, _acc = sess.run([loss_op, accuracy_op], feed_dict=feed_dict)

        loss += _loss * args.batch_size
        acc += _acc * args.batch_size

    str = '{} {:.4f} {:.4f}'.format(epoch, loss / num_images, acc / num_images)
    print(str)


def main(args):
    if args.eagerFlag in [1, 2]:
        tfe.enable_eager_execution()  # Enable eager mode. Once activated it cannot be reversed! Run just once.
    if args.phase == 'train':
        if args.eagerFlag==0:
            trainNoEager(args)
        elif args.eagerFlag==1:
            trainEagerV1(args)
        else:
            trainEagerV2(args)
    else:

        if args.eagerFlag==0:
            testNoEager(args)
        elif args.eagerFlag==1:
            testEagerV1(args)
        else:
            testEagerV2(args)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for storing input data')  # data
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help='Training or test phase, should be one of {"train", "test"}')
    parser.add_argument('--lr', type=float, default=[0.1],nargs='+',  help='')

    parser.add_argument('--batch_size', type=int, default=100, help='')
    parser.add_argument('--epoch', type=int, default=1000, help='')  #
    parser.add_argument('--resultFolder', type=str, default='result', help='')
    parser.add_argument('--save_step', type=int, default=200, help='')
    parser.add_argument('--saveMaxModelNum', type=int, default=3, help='')
    parser.add_argument('--eagerFlag', type=int, default=2, choices=[0, 1, 2], help='0: do not use eager, 1: use eager v1, 2: use eager v2')

    parser.add_argument('--classNum', type=int, default=10, help='')
    parser.add_argument('--height', type=int, default=28, help='')
    parser.add_argument('--width', type=int, default=28, help='')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
