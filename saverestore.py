from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import tensorflow.contrib.eager as tfe



def restoreModelV1(model_dir, model):
    dummy_input = tf.constant(tf.zeros((1, 28, 28, 1)))  # Run the model once to initialize variables
    dummy_pred = model(dummy_input, training=False)

    saver = tfe.Saver(model.variables)  # Restore the variables of the model
    saver.restore(tf.train.latest_checkpoint(model_dir))


def saveModelV1(model_dir, model, global_step, modelname='model1'):
    tfe.Saver(model.variables).save(os.path.join(model_dir, modelname), global_step=global_step)


def restoreModelV2(model_dir, checkpoint):
    checkpoint.restore(tf.train.latest_checkpoint(model_dir))


def saveModelV2(model_dir, checkpoint, modelname='model2'):
    checkpoint_prefix = os.path.join(model_dir, modelname)
    checkpoint.save(checkpoint_prefix)


def restoreModelNoEager(sess, saver, model_dir):
    ckpt = tf.train.latest_checkpoint(model_dir)
    if ckpt:
        saver.restore(sess, ckpt)


def saveModelNoEager(sess, saver, model_dir, model_name, step):
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        saver.export_meta_graph(metagraph_filename)
    summary = tf.Summary()
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)