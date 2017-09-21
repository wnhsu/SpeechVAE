"""Template runner for training and testing VAE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import truncnorm
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import os
import time
import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict, defaultdict

from utils import *
from runners import *
from tools.vis import plot_rows
from kaldi_io import BaseFloatMatrixWriter as BFMWriter
from kaldi_io import BaseFloatVectorWriter as BFVWriter
from kaldi_io import SequentialBaseFloatVectorReader as SBFVReader

DEFAULT_WRITE_FN = lambda uttid, feats: print("%s\n%s" % (uttid, feats))

def train(exp_dir, model, train_set, dev_set, train_conf, n_print_steps=200):
    if os.path.exists("%s/.done" % exp_dir):
        info("training is already done. exit..")
        return

    dev_iterator_fn     = lambda: dev_set.iterator(2048) if dev_set else None 
    n_epochs            = train_conf.pop("n_epochs")
    n_patience          = train_conf.pop("n_patience")
    bs                  = train_conf.pop("bs")
    n_steps_per_epoch   = train_conf.pop("n_steps_per_epoch")
    assert(n_steps_per_epoch > 0)

    model_dir = "%s/models" % exp_dir
    check_and_makedirs(model_dir)
    ckpt_path = os.path.join(model_dir, "vae.ckpt")

    # create summaries
    sum_names = ["loss", "lb", "neg_kld", "logpx_z"]
    sum_vars = [tf.reduce_mean(model.outputs[name]) for name in sum_names]

    with tf.variable_scope("train"):
        train_summaries = tf.summary.merge(
                [tf.summary.scalar(*p) for p in zip(sum_names, sum_vars)])

    with tf.variable_scope("test"):
        test_vars = OrderedDict([(name, tf.get_variable(name, initializer=0.)) \
                for name in sum_names])
        test_summaries = tf.summary.merge(
                [tf.summary.scalar(k, test_vars[k]) for k in test_vars])

    def make_feed_dict(inputs, targets, is_train):
        feed_dict = {model.feed_dict["inputs"]: inputs,
                     model.feed_dict["targets"]: targets,
                     model.feed_dict["masks"]: np.ones_like(inputs),
                     model.feed_dict["is_train"]: is_train}
        return feed_dict

    global_step = -1
    epoch = -1
    passes = 0     # number of dataset passes this run
    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        init_step = model.init_or_restore_model(sess, model_dir)
        global_step = int(init_step)
        epoch = int(global_step // n_steps_per_epoch)
        info("init or restore model takes %.2f s" % (time.time() - start_time))
        info("current #steps=%s, #epochs=%s" % (global_step, epoch))
        if epoch >= n_epochs:
            info("training is already done. exit..")
            return
        train_writer = tf.summary.FileWriter("%s/log/train" % exp_dir, sess.graph)
        dev_writer = tf.summary.FileWriter("%s/log/dev" % exp_dir)

        info("start training...")
        best_epoch = -1
        best_dev_lb = -np.inf
        train_start_time = time.time()
        print_start_time = time.time()

        while True:
            for inputs, _, _, targets in train_set.iterator(bs):
                global_step, _ = sess.run(
                        [model.global_step, model.ops["train_step"]], 
                        make_feed_dict(inputs, targets, 1))

                if global_step % n_print_steps == 0 and global_step != init_step:
                    outputs, summary = sess.run(
                            [sum_vars, train_summaries], 
                            make_feed_dict(inputs, targets, 0))
                    train_writer.add_summary(summary, global_step)
                    info("[epoch %.f step %.f pass %.f]: " % (
                                epoch, global_step, passes) + \
                            "print time=%.2fs" % (
                                time.time() - print_start_time) + \
                            ", total time=%.2fs, " % (
                                time.time() - train_start_time) + \
                            ", ".join(["%s %.4f" % p for p in zip(
                                sum_names, outputs)]))
                    print_start_time = time.time()
                    if np.isnan(outputs[0]):
                        info("...exit training and not saving this epoch")
                        return

                if global_step % n_steps_per_epoch == 0 and global_step != init_step:
                    if dev_iterator_fn:
                        val_start_time = time.time()
                        dev_vals = _valid(sess, model, sum_names, sum_vars, dev_iterator_fn)
                        feed_dict = dict(zip(test_vars.values(), dev_vals.values()))
                        summary = sess.run(test_summaries, feed_dict)
                        dev_writer.add_summary(summary, global_step)
                        info("[epoch %.f]: dev  \t" % epoch + \
                                "valid time=%.2fs" % (
                                    time.time() - val_start_time) + \
                                ", total time=%.2fs, " % (
                                    time.time() - train_start_time) + \
                                ", ".join(["%s %.4f" % p for p in dev_vals.items()]))
                        if dev_vals["lb"] > best_dev_lb:
                            best_epoch, best_dev_lb = epoch, dev_vals["lb"]
                            model.saver.save(sess, ckpt_path)
                        if epoch - best_epoch > n_patience:
                            info("...running out of patience" + \
                                    ", time elapsed=%.2fs" % (
                                        time.time() - train_start_time) + \
                                    ", exit training")
                            open("%s/.done" % exp_dir, "a")
                            return

                    epoch += 1
                    if epoch >= n_epochs:
                        info("...finish training" + \
                                ", time elapsed=%.2fs" % (
                                    time.time() - train_start_time))
                        open("%s/.done" % exp_dir, "a")
                        return
                    
            passes += 1

def test(exp_dir, model, test_set):
    test_iterator_fn    = lambda: test_set.iterator(2048) 

    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "vae.ckpt")
    
    # create summaries
    sum_names = ["loss", "lb", "neg_kld", "logpx_z"]
    sum_vars = [tf.reduce_mean(model.outputs[name]) for name in sum_names]

    with tf.variable_scope("test"):
        test_vars = OrderedDict([(name, tf.get_variable(name, initializer=0.)) \
                for name in sum_names])
        test_summaries = tf.summary.merge(
                [tf.summary.scalar(k, test_vars[k]) for k in test_vars])

    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))
        test_writer = tf.summary.FileWriter("%s/log/test" % exp_dir)

        test_vals = _valid(sess, model, sum_names, sum_vars, test_iterator_fn)
        feed_dict = dict(zip(test_vars.values(), test_vals.values()))
        summary, global_step = sess.run([test_summaries, model.global_step], feed_dict)
        test_writer.add_summary(summary, global_step)
        info("test\t" + ", ".join(["%s %.4f" % p for p in test_vals.items()]))

def dump_repr(
        exp_dir, model, dataset, repr_set_name, is_talabel,
        label_to_str, write_fn=DEFAULT_WRITE_FN):
    if is_talabel:
        iterator_fn = lambda: dataset.talabel_iterator(2048, repr_set_name)
    else:
        iterator_fn = lambda: dataset.iterator(2048, repr_set_name)

    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "vae.ckpt")

    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))
        
        label2repr = _est_repr(sess, model, iterator_fn, debug=True)

        for label in sorted(label2repr.keys()):
            debug("writing %s" % label_to_str[label])
            write_fn(label_to_str[label], label2repr[label])

    info("...finish dumping representations, time elapsed=%.2fs" % (time.time() - start_time))

def repl_repr_utt(
        exp_dir, model, dataset, repr_set_name, label_to_str, 
        label_str_to_repr, repl_list, write_fn=DEFAULT_WRITE_FN):
    """modify the attribute of selected labels
    feat_rspec set to test_feat_rspec by default
    label for mod_label_in is repr_label (str)"""
    iterator_by_utt_str_fn = \
            lambda utt_str: dataset.iterator_by_label(
                    2048, "uttid", dataset.get_label("uttid", utt_str))
    
    model_dir = "%s/models" % exp_dir
    ckpt_path = os.path.join(model_dir, "vae.ckpt")
  
    with tf.Session(config=SESS_CONF) as sess:
        start_time = time.time()
        model.saver.restore(sess, ckpt_path)
        info("restore model takes %.2f s" % (time.time() - start_time))
        
        start_time = time.time()
        for i, (utt_str, label_str) in enumerate(repl_list):
            utt_feats = np.concatenate(
                    [inputs for inputs, _, _, _ in iterator_by_utt_str_fn(utt_str)],
                    axis=0)
            src_label_str = label_to_str[dataset.get_label(repr_set_name, utt_str)]
            src_repr = label_str_to_repr[src_label_str]
            tar_repr = label_str_to_repr[label_str]
            debug("utt: %s, shape %s" % (utt_str, utt_feats.shape) + \
                    ", %s label: %s => %s" % (repr_set_name, src_label_str, label_str))
            Z, Z_mod, utt_feats_mod = _replace_repr(
                    sess, model, utt_feats, src_repr, tar_repr)
            raw_utt_feats_mod = dataset.undo_mvn(dataset.target_to_feat(utt_feats_mod))
            utt_str_mod = "%s_modto_%s" % (utt_str, label_str)

            # NOTES: except for the first segment, use only last frame
            raw_utt_segs_mod = [raw_utt_feats_mod[0]]    # (C, seg_len, F)
            for raw_utt_seg_mod in raw_utt_feats_mod[1:]:
                raw_utt_segs_mod.append(raw_utt_seg_mod[:, -1:, :])
            nonoverlap_raw_utt_feats_mod = np.concatenate(raw_utt_segs_mod, axis=-2)
            write_fn(utt_str_mod, nonoverlap_raw_utt_feats_mod)

    info("...finish replacing representations, time elapsed=%.2fs" % (time.time() - start_time))

def _valid(sess, model, sum_names, sum_vars, iterator_fn):
    vals = OrderedDict([(name, 0) for name in sum_names])
    n_batches = 0
    for inputs, _, _, targets in iterator_fn():
        n_batches += 1
        outputs = sess.run(
                sum_vars, 
                feed_dict={
                    model.feed_dict["inputs"]: inputs,
                    model.feed_dict["targets"]: targets,
                    model.feed_dict["masks"]: np.ones_like(inputs),
                    model.feed_dict["is_train"]: 0})
        for name, val in zip(sum_names, outputs):
            vals[name] += val
    for name in vals:
        vals[name] /= n_batches
    return vals

def _encode_z_mean_fn(sess, model, inputs):
    feed_dict = {model.feed_dict["inputs"]: inputs,
                 model.feed_dict["is_train"]: 0}
    return sess.run(model.outputs["qz_x"][0], feed_dict)

def _decode_x_mean_fn(sess, model, Z):
    feed_dict = {model.outputs["z"]: Z,
                 model.feed_dict["is_train"]: 0}
    return sess.run(model.outputs["px_z"][0], feed_dict)

def _est_repr(sess, model, iterator_fn, debug=False):
    # ML estimation of repr for dataset
    label_to_acc_z = defaultdict(float)
    label_to_N = defaultdict(float)
    label_to_repr = dict()
    start_time = time.time()
    for inputs, _, labels, _ in iterator_fn():
        Z = _encode_z_mean_fn(sess, model, inputs)
        for z, label in zip(Z, labels):
            label_to_acc_z[label] += z
            label_to_N[label] += 1
    for label in label_to_N:
        label_to_repr[label] = label_to_acc_z[label] / label_to_N[label]
    if debug:
        for label in sorted(label_to_repr):
            info("length=%.2f, N=%s, label=%s" % ( 
                np.linalg.norm(label_to_repr[label]),
                label_to_N[label], label,))
    return label_to_repr

def _replace_repr(sess, model, inputs, src_repr, tar_repr):
    Z = _encode_z_mean_fn(sess, model, inputs)
    Z_mod = Z - src_repr[np.newaxis, :] + tar_repr[np.newaxis, :]
    inputs_mod = _decode_x_mean_fn(sess, model, Z_mod)
    return Z, Z_mod, inputs_mod
