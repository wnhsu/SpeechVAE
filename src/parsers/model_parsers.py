#!/usr/bin/python 

from parser_common import *
from libs.activations import *

nl_dict = {"None": None,
           "tanh": tf.nn.tanh,
           "relu": tf.nn.relu,
           "sigmoid": tf.nn.sigmoid,
           "relu_n10": lambda x: custom_relu(x, cutoff=-10)}
inv_nl_dict = dict(zip(nl_dict.values(), nl_dict.keys()))

def parse_raw_fc_str(raw_fc_str):
    return [int(hu) for hu in raw_fc_str.split(',') if hu]

def fc_conf_to_str(fc_conf):
    return ','.join(map(str, fc_conf))

def parse_raw_conv_str(raw_conv_str):
    """
    raw_conv_str is: 
        #FILTERS,KERNEL_HEIGHT,KERNEL_WIDTH[,STRIDE_HEIGHT,STRIDE_WIDTH[,PADDING]]
    """
    conv = []
    raw_layers = [l.split('_') for l in raw_conv_str.split(',') if l]
    for raw_layer in raw_layers:
        if len(raw_layer) == 3:
            conv.append(tuple([int(x) for x in raw_layer]) + (1, 1, "valid"))
        elif len(raw_layer) == 5:
            conv.append(tuple([int(x) for x in raw_layer]) + ("same",))
        elif len(raw_layer) == 6:
            conv.append(tuple([int(x) for x in raw_layer[:5]]+[raw_layer[5]]))
        else:
            raise ValueError("raw_layer format invalid: %s" % str(raw_layer))
    return conv

def conv_conf_to_str(conv_conf):
    return ",".join(["_".join(map(str, layer)) for layer in conv_conf])

class base_model_parser(base_parser):
    def __init__(self, model_config_path):
        self.parser = DefaultConfigParser()

        parser = self.parser
        config = {}
        if len(parser.read(model_config_path)) == 0:
            raise ValueError("base_model_parser(): %s not found", model_config_path)

        config["input_shape"]   = None
        config["target_shape"]  = None
        config["n_latent"]      = parser.getint("model", "n_latent")
        config["x_conti"]       = parser.getboolean("model", "x_conti")
        config["x_mu_nl"]       = None
        config["x_logvar_nl"]   = None 
        config["n_bins"]        = None
        config["if_bn"]         = parser.getboolean("model", "if_bn", True)

        if config["x_conti"]:
            config["x_mu_nl"] = nl_dict[parser.get("model", "x_mu_nl")]
            config["x_logvar_nl"] = nl_dict[parser.get("model", "x_logvar_nl")]
        else:
            config["n_bins"] = parser.getint("model", "n_bins")
        
        self.config = config
    
    @staticmethod
    def write_config(config, f):
        f.write("[model]\n")
        for key in ["n_latent", "if_bn", "x_conti"]:
            f.write("%s= %s\n" % (key.ljust(20), str(config[key])))
        if config["x_conti"]:
            for key in ["x_mu_nl", "x_logvar_nl"]:
                f.write("%s= %s\n" % (key.ljust(20), str(inv_nl_dict[config[key]])))
        else:
            for key in ["n_bins"]:
                f.write("%s= %s\n" % (key.ljust(20), str(config[key])))

class fvae_model_parser(base_model_parser):
    def __init__(self, model_config_path):
        super(fvae_model_parser, self).__init__(model_config_path)

        self.config["hu_enc"]   = parse_raw_fc_str(
                self.parser.get("model", "hu_enc", ""))

    def get_config(self):
        return self.config

    @staticmethod
    def write_config(config, f):
        super(fvae_model_parser, fvae_model_parser).write_config(config, f)
        f.write("\n")
        for key in ["hu_enc"]:
            f.write("%s= %s\n" % (key.ljust(20), fc_conf_to_str(config[key])))

class cvae_model_parser(fvae_model_parser):
    def __init__(self, model_config_path):
        super(cvae_model_parser, self).__init__(model_config_path)

        self.config["conv_enc"] = parse_raw_conv_str(
                self.parser.get("model", "conv_enc", ""))

    @staticmethod
    def write_config(config, f):
        super(cvae_model_parser, cvae_model_parser).write_config(config, f)
        f.write("\n")
        for key in ["conv_enc"]:
            f.write("%s= %s\n" % (key.ljust(20), conv_conf_to_str(config[key])))
