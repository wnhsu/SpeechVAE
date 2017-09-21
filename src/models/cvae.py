"""Convolutional Variational Autoencoder Class"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import nn
from tensorflow.contrib.layers import fully_connected, conv2d, conv2d_transpose
from tensorflow.contrib.layers import batch_norm, l2_regularizer

from libs import get_conv_output_shape
from libs.layers import dense_latent, deconv_latent
from models import *
from models.base_vae import BaseVAE

class CVAE(BaseVAE):
    """
    Convolutional + Fully-Connected VAE
    """
    def __init__(self, model_conf, train_conf, **kwargs):
        super(CVAE, self).__init__(model_conf, train_conf)

    def _set_model_conf(self, model_conf, sym=True):
        """
        Args:
            - model_conf: specifies the model configurations.
                - input_shape: list of (c, h, w)
                - input_dtype: input data type
                - target_shape: list of (c, h, w)
                - target_dtype: target data type
                - conv_enc: list of conv layer specs for encoder
                    (c, k_h, k_w, s_h, s_w, padding)
                - hu_enc: list of int of number of hidden 
                    units at each layer for encoder
                - hu_dec: list of int of number of hidden 
                    units at each layer for decoder. do not
                    specify if using symmetric architecture
                - deconv_dec: list of deconv layer specs for decoder
                    (c, k_h, k_w, s_h, s_w, padding)
                - n_latent: number of latent variables 
                - x_conti: whether target is continuous or discrete
                    use Gaussian for continuous target, softmax
                    for discrete target
                - x_mu_nl: activation function for target mean
                - x_logvar_nl: activation function for target 
                    log variance
                - n_bins: discretized target dimension
                - if_bn: use batch normalization if True
            - sym: True if using symmetric architecture
        """
        self._model_conf = {"input_shape": None,
                            "input_dtype": tf.float32,
                            "target_shape": None,
                            "target_dtype": tf.float32,
                            "conv_enc": [],
                            "conv_enc_output_shape": None,
                            "hu_enc": [],
                            "hu_dec": [],
                            "deconv_dec": [],
                            "n_latent": 128,
                            "x_conti": True,
                            "x_mu_nl": None,
                            "x_logvar_nl": None,
                            "n_bins": None,
                            "if_bn": True,
                            "sym": sym}
        for k in model_conf:
            if k in self._model_conf:
                self._model_conf[k] = model_conf[k]
            else:
                raise ValueError("invalid model_conf: %s" % str(k))
        
        if not sym:
            return
        if not self._model_conf["conv_enc"]:
            raise ValueError("need at least one Convolutional layer")
        if self._model_conf["hu_dec"]:
            raise ValueError("do not specify hu_dec if using symmetric model")
        if self._model_conf["deconv_dec"]:
            raise ValueError("do not specify deconv_dec if using symmetric model")

        self._model_conf["hu_dec"] = self._model_conf["hu_enc"][::-1]
        # add a dense layer match the last conv output dim
        conv_shape = get_conv_output_shape(
                self._model_conf["input_shape"], 
                self._model_conf["conv_enc"])
        self._model_conf["hu_dec"].append(np.prod(conv_shape))
        self._model_conf["conv_enc_output_shape"] = conv_shape

        # compute deconv layer shapes
        for i in range(len(self._model_conf["conv_enc"]) - 1)[::-1]:
            self._model_conf["deconv_dec"].append(
                    self._model_conf["conv_enc"][i][:1] + \
                    self._model_conf["conv_enc"][i + 1][1:])
        
        info("MODEL CONFIG:\n%s" % str(self._model_conf))
        
    def _build_encoder(self, inputs, reuse=False):
        weights_regularizer = l2_regularizer(self._train_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}

        outputs = inputs
        with tf.variable_scope("enc"):
            for i, (c, h, w, s_h, s_w, pad) in \
                    enumerate(self._model_conf["conv_enc"]):
                outputs = conv2d(
                        inputs=outputs,
                        num_outputs=c,
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        activation_fn=nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="enc_conv%s" % (i + 1))

            output_dim = np.prod(outputs.get_shape().as_list()[1:])
            outputs = tf.reshape(outputs, [-1, output_dim])

            for i, hu in enumerate(self._model_conf["hu_enc"]):
                outputs = fully_connected(
                        inputs=outputs,
                        num_outputs=hu,
                        activation_fn=nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="enc_fc%s" % (i + 1))

            z_mu, z_logvar, z = dense_latent(outputs,
                                             self._model_conf["n_latent"],
                                             scope="latent_z")
        return [z_mu, z_logvar], z

    def _build_decoder(self, inputs, reuse=False):
        weights_regularizer = l2_regularizer(self._train_conf["l2_weight"])
        normalizer_fn = batch_norm if self._model_conf["if_bn"] else None
        normalizer_params = None
        if self._model_conf["if_bn"]:
            normalizer_params = {"is_training": self._feed_dict["is_train"], 
                                 "reuse": reuse}

        outputs = inputs
        with tf.variable_scope("dec"):
            for i, hu in enumerate(self._model_conf["hu_dec"]):
                outputs = fully_connected(
                        inputs=outputs,
                        num_outputs=hu,
                        activation_fn=nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="dec_fc%s" % (i + 1))
            
            # if no deconvolutional layers, use dense_latent for target
            target_shape = list(self._model_conf["target_shape"])
            mu_nl = self._model_conf["x_mu_nl"]
            logvar_nl = self._model_conf["x_logvar_nl"]
            outputs = tf.reshape(
                    outputs, 
                    (-1,) + self._model_conf["conv_enc_output_shape"])
            for i, (c, h, w, s_h, s_w, pad) in \
                    enumerate(self._model_conf["deconv_dec"]):
                outputs = conv2d_transpose(
                        inputs=outputs,
                        num_outputs=c,
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        activation_fn=nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=weights_regularizer,
                        reuse=reuse,
                        scope="dec_deconv%s" % (i + 1))

            h, w, s_h, s_w, pad = self._model_conf["conv_enc"][0][1:]
            post_trim = (slice(None, target_shape[1]), slice(None, target_shape[2]))
            if self._model_conf["x_conti"]:
                x_mu, x_logvar, x = deconv_latent(
                        outputs,
                        num_outputs=target_shape[0],
                        kernel_size=(h, w),
                        stride=(s_h, s_w),
                        padding=pad,
                        data_format=DATA_FORMAT,
                        mu_nl=mu_nl,
                        logvar_nl=logvar_nl,
                        post_trim=post_trim,
                        scope="recon_x")
                px_z = [x_mu, x_logvar]
            else:
                raise NotImplementedError()

        return px_z, x
