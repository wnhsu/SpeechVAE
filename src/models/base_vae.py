"""Base Variational Autoencoder Class"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from models import *
from libs.costs import kld, log_gauss
sce_logits = tf.nn.sparse_softmax_cross_entropy_with_logits

class BaseVAE(object):
    """
    Abstract class for VAE, should never be called directly.
    """
    def __init__(self, model_conf, train_conf):
        # create data members
        self._model_conf    = None
        self._train_conf    = None

        self._feed_dict     = None  # feed dict needed for outputs
        self._layers        = None  # outputs at each layer
        self._outputs       = None  # general outputs (acc, posterior...)
        self._global_step   = None  # global_step for saver

        self._ops           = None  # accessible ops (train_step, decay_op...)

        # set model conf
        self._set_model_conf(model_conf)

        # set train conf
        self._set_train_conf(train_conf)

        # build model
        self._build_model()

        # create saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        debug("Created saver for these variables:\n%s" % str(
                [p.name for p in tf.global_variables()]))
    
    @property
    def model_conf(self):
        return self._model_conf

    @property
    def train_conf(self):
        return self._train_conf

    @property
    def feed_dict(self):
        return self._feed_dict

    @property
    def outputs(self):
        return self._outputs

    @property
    def global_step(self):
        return self._global_step

    @property
    def ops(self):
        return self._ops

    def _set_model_conf(self, model_conf):
        # minimum schema for VAE models
        self._model_conf = {"input_shape": None,
                            "input_dtype": tf.float32,
                            "target_shape": None,
                            "target_dtype": tf.float32,
                            "n_latent": 128,
                            "x_conti": True,
                            "x_mu_nl": None,
                            "x_logvar_nl": None,
                            "n_bins": None}
        raise NotImplementedError

    def _set_train_conf(self, train_conf):
        # training schema
        self._train_conf = {"lr": 0.001,
                            "lr_decay_factor": 0.8,
                            "l2_weight": 0.0001,
                            "max_grad_norm": None,
                            "opt": "adam",
                            "opt_opts": {}}
        for k in train_conf:
            if k in self._train_conf:
                self._train_conf[k] = train_conf[k]
            else:
                info("WARNING: unused train_conf: %s" % str(k))
        
        for k in ["lr", "lr_decay_factor", "l2_weight", "max_grad_norm"]:
            if self._train_conf[k] is not None:
                self._train_conf[k] = tf.get_variable(
                        k, trainable=False, initializer=float(self._train_conf[k]))

    def _build_model(self):
        # create placeholders
        inputs = tf.placeholder(self._model_conf["input_dtype"],
                                shape=(None,)+self._model_conf["input_shape"],
                                name="inputs")
        targets = tf.placeholder(self._model_conf["target_dtype"],
                                 shape=(None,)+self._model_conf["target_shape"],
                                 name="targets")
        masks = tf.placeholder(tf.float32,
                               shape=(None,)+self._model_conf["target_shape"],
                               name="masks")
        is_train = tf.placeholder(tf.bool, name="is_train")
        self._feed_dict = {"inputs": inputs,
                           "targets": targets,
                           "masks": masks,
                           "is_train": is_train}

        # build encoder/decoder and outputs
        # qz_x/px_z are distribution paramters, 
        # [mu, logvar] for gaussian, 
        # logits of shape x.shape + (n_bins,) for discrete
        qz_x, z = self._build_encoder(inputs)
        px_z, x = self._build_decoder(z)
        
        with tf.name_scope("costs"):
            with tf.name_scope("neg_kld"):
                neg_kld = tf.reduce_mean(
                        tf.reduce_sum(-1 * kld(*qz_x), axis=1))

            masks = self._feed_dict["masks"]
            targets = self._feed_dict["targets"]
            with tf.name_scope("logpx_z"):
                if self._model_conf["x_conti"]:
                    info("px_z is gaussian")
                    x_mu, x_logvar = px_z
                    logpx_z = tf.reduce_mean(tf.reduce_sum(
                            masks * log_gauss(x_mu, x_logvar, targets), 
                            axis=(1, 2, 3)))
                else:
                    raise NotImplementedError()

            with tf.name_scope("l2"):
                reg_loss = tf.reduce_sum(tf.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES))

            lb = neg_kld + logpx_z
            loss = -lb + reg_loss
        
        self._outputs = {"qz_x": qz_x, "z": z, "px_z": px_z, "x": x,
                         "neg_kld": neg_kld, "logpx_z": logpx_z, 
                         "reg_loss": reg_loss, "lb": lb, "loss": loss}
        
        # create ops for training
        self._build_train()
        
    def _build_train(self):
        # create grads and clip optionally
        params = tf.trainable_variables()
        self._global_step = tf.get_variable(
                "global_step", trainable=False, initializer=0.0)
        
        with tf.name_scope("grad"):
            grads = tf.gradients(self._outputs["loss"], params)
            if self._train_conf["max_grad_norm"] is None:
                clipped_grads = grads
            else:
                clipped_grads, _ = tf.clip_by_global_norm(
                        grads, self._train_conf["max_grad_norm"])

        # create ops
        with tf.name_scope("train"):
            lr = self._train_conf["lr"]
            opt = self._train_conf["opt"]
            opt_opts = self._train_conf["opt_opts"]

            if opt == "adam":
                info("Using Adam as the optimizer")
                opt = tf.train.AdamOptimizer(learning_rate=lr, **opt_opts)
            elif opt == "sgd":
                info("Using SGD as the optimizer")
                opt = tf.train.GradientDescentOptimizer(learning_rate=lr, **opt_opts)
            else:
                raise ValueError("optimizer %s not supported" % opt)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = opt.apply_gradients(zip(clipped_grads, params),
                                                 global_step=self._global_step)
            
            decay_op = lr.assign(lr * self._train_conf["lr_decay_factor"])

        self._ops = {"train_step": train_step, "decay_op": decay_op}
        
    def _build_encoder(self):
        """Return qz_x, z"""
        raise NotImplementedError

    def _build_decoder(self):
        """Return px_z, x"""
        raise NotImplementedError

    def init_or_restore_model(self, sess, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            info("Reading model params from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            info("Creating model with fresh params")
            sess.run(tf.global_variables_initializer())
        return sess.run(self._global_step)
