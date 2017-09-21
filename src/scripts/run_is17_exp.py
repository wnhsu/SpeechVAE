import os
from runners import *
from runners.vae_runner import train, test, dump_repr, repl_repr_utt
from tools.kaldi.prep_kaldi_feat import flatten_channel
from datasets.datasets_loaders import datasets_loader
from datasets.datasets_loaders import get_frame_ra_dataset_conf
from parsers.train_parsers import vae_train_parser as train_parser
from parsers.dataset_parsers import kaldi_ra_dataset_parser as dataset_parser
from tools.vis import plot_heatmap
from tools.kaldi.plot_scp import plot_kaldi_feat
from kaldi_io import BaseFloatMatrixWriter as BFMWriter
from kaldi_io import BaseFloatVectorWriter as BFVWriter
from kaldi_io import SequentialBaseFloatVectorReader as SBFVReader

# flags for all actions
tf.app.flags.DEFINE_string("arch", "cnn", "dnn|cnn|rnn")
tf.app.flags.DEFINE_string("exp_dir", "", "path to dump/load this experiment")

# flags for if running optional actions
tf.app.flags.DEFINE_boolean("test", True, "run testing if True")
tf.app.flags.DEFINE_boolean("dump_spk_repr", False, "dump latent speaker representations if True")
tf.app.flags.DEFINE_boolean("dump_phone_repr", False, "dump latent phone representations if True")
tf.app.flags.DEFINE_boolean("comp_ortho", False, "dump latent phone representations if True")
tf.app.flags.DEFINE_boolean("repl_repr_utt", False, "dump latent variable by frame if True")

# flags for train
tf.app.flags.DEFINE_string("train_conf", "", "path to training config")
tf.app.flags.DEFINE_string("model_conf", "", "path to model config")
tf.app.flags.DEFINE_string("dataset_conf", "", "path to dataset config")
tf.app.flags.DEFINE_integer("n_print_steps", 200, "print training diagnostics every N steps")

# flags for non-train actions
tf.app.flags.DEFINE_string("feat_rspec", "", "feat rspecifier to replace test_feat_rspec")
tf.app.flags.DEFINE_string("feat_set_name", "", "name(s) of label set(s)")
tf.app.flags.DEFINE_string("feat_utt2label_path", "", "utt2label_path(s) for feat_rspec")
tf.app.flags.DEFINE_string("feat_label_N", "", "number(s) of classes for feat_rspec")
tf.app.flags.DEFINE_string("feat_ta_set_name", "", "name(s) of label set(s)")
tf.app.flags.DEFINE_string("feat_utt2talabels_path", "", "utt2talabels_path(s) for feat_rspec")
tf.app.flags.DEFINE_string("feat_talabel_N", "", "number(s) of talabel classes for feat_rspec")
 
# flags for dump_repr (spk)
tf.app.flags.DEFINE_string("spk_repr_spec", "", "test set attribute representation read/write specifier")
tf.app.flags.DEFINE_string("spk_repr_id_map", "", "path to label id map to dump")

# flags for dump_repr (phone)
tf.app.flags.DEFINE_string("phone_repr_spec", "", "test set attribute representation read/write specifier")
tf.app.flags.DEFINE_string("phone_repr_id_map", "", "path to label id map to dump")

# flags for comp_ortho
# --spk_repr_spec/--phone_repr_spec
tf.app.flags.DEFINE_string("ortho_img_path", "", "absolute cos-sim plot path")

# flags for repl_repr_utt
# --repr_spec/--repr_set_name/--repr_id_map
tf.app.flags.DEFINE_string("repl_utt_wspec", "", "test set replacing representation write specifier")
tf.app.flags.DEFINE_string("repl_utt_list", "", "test set replacing representation list: (uttid_str, attr_str)")
tf.app.flags.DEFINE_string("repl_utt_img_dir", "", "test set replacing representation image directory")

FLAGS = tf.app.flags.FLAGS

def main():
    if FLAGS.arch == "dnn":
        raise NotImplementedError()
    elif FLAGS.arch == "cnn":
        from models.cvae import CVAE as model_class
        from parsers.model_parsers import cvae_model_parser as model_parser
    elif FLAGS.arch == "rnn":
        raise NotImplementedError()
    else:
        raise ValueError("unsupported architecture %s" % FLAGS.arch)
    
    # do training
    if not os.path.exists("%s/.done" % FLAGS.exp_dir):
        set_logger(custom_logger("%s/log/train.log" % FLAGS.exp_dir))
        is_train = True
        exp_dir, model_conf, train_conf, dataset_conf = _load_configs(
                FLAGS, model_parser, is_train)
        _print_configs(exp_dir, model_conf, train_conf, dataset_conf)
        [train_set, dev_set, _], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, 
                is_train, True, True, False)
        train(exp_dir, model, train_set, dev_set, train_conf, FLAGS.n_print_steps)
        unset_logger()
    
    # do testing
    if FLAGS.test:
        set_logger(custom_logger("%s/log/test.log" % FLAGS.exp_dir))
        is_train = False
        tf.reset_default_graph()
        exp_dir, model_conf, train_conf, dataset_conf = _load_configs(
                FLAGS, model_parser, is_train)
        _print_configs(exp_dir, model_conf, train_conf, dataset_conf)
        [_, _, test_set], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, 
                is_train, False, False, True)
        test(exp_dir, model, test_set)
        unset_logger()

    # do dumping speaker representations
    if FLAGS.dump_spk_repr:
        set_logger(custom_logger("%s/log/dump_spk_repr.log" % FLAGS.exp_dir))
        is_train = False
        is_talabel = False
        tf.reset_default_graph()
        exp_dir, model_conf, train_conf, dataset_conf, \
                spk_repr_set_name, spk_repr_id_map, spk_repr_spec = \
                _load_dump_spk_repr_configs(FLAGS, model_parser)

        _print_configs(exp_dir, model_conf, train_conf, dataset_conf)
        info("Dump Representation Configurations:\n\tset_name: %s, wspec: %s" % (
                spk_repr_set_name, spk_repr_spec))
        [_, _, test_set], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, 
                is_train, False, False, True)

        for path in spk_repr_spec.split(":")[1].split(","):
            check_and_makedirs(os.path.dirname(path))
        with BFVWriter(spk_repr_spec) as writer:
            dump_repr(
                    exp_dir, model, test_set, spk_repr_set_name, 
                    is_talabel, spk_repr_id_map, writer.write)
        unset_logger()
    
    # do dumping phone representations
    if FLAGS.dump_phone_repr:
        set_logger(custom_logger("%s/log/dump_phone_repr.log" % FLAGS.exp_dir))
        is_train = False
        is_talabel = True
        tf.reset_default_graph()
        exp_dir, model_conf, train_conf, dataset_conf, \
                phone_repr_set_name, phone_repr_id_map, phone_repr_spec = \
                _load_dump_phone_repr_configs(FLAGS, model_parser)

        _print_configs(exp_dir, model_conf, train_conf, dataset_conf)
        info("Dump Representation Configurations:\n\tset_name: %s, wspec: %s" % (
                phone_repr_set_name, phone_repr_spec))
        [_, _, test_set], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, 
                is_train, False, False, True)

        for path in phone_repr_spec.split(":")[1].split(","):
            check_and_makedirs(os.path.dirname(path))
        with BFVWriter(phone_repr_spec) as writer:
            dump_repr(
                    exp_dir, model, test_set, phone_repr_set_name, 
                    is_talabel, phone_repr_id_map, writer.write)
        unset_logger()

    # do compute representation absolute cosine similarities
    if FLAGS.comp_ortho:
        set_logger(custom_logger("%s/log/comp_ortho.log" % FLAGS.exp_dir))
        exp_dir, ortho_img_path, spk_repr_spec, phone_repr_spec = \
                _load_comp_ortho_configs(FLAGS, model_parser)
        info("Compute Latent Attribute Representation Absolute Cosine Similarities")
        info("\tspeaker repr: %s\n\tphone repr: %s" % (spk_repr_spec, phone_repr_spec))
        _comp_abs_cos(ortho_img_path, spk_repr_spec, phone_repr_spec)
        unset_logger()

    # do replacing representations
    if FLAGS.repl_repr_utt:
        set_logger(custom_logger("%s/log/repl_repr_utt.log" % FLAGS.exp_dir))
        is_train = False
        tf.reset_default_graph()
        exp_dir, model_conf, train_conf, dataset_conf, \
                repr_set_name, repr_spec, repr_id_map, label_str_to_repr, \
                repl_utt_list, repl_utt_wspec, repl_utt_img_dir = \
                _load_repl_spk_repr_utt_configs(FLAGS, model_parser)

        _print_configs(exp_dir, model_conf, train_conf, dataset_conf)
        info("Replacing Representation Configurations:")
        info("\tset_name: %s, rspec: %s" % (repr_set_name, repr_spec))
        info("\trepl_utt_list[0]: %s, repl_utt_wspec: %s" % (repl_utt_list[0], repl_utt_wspec))
        [_, _, test_set], model = _load_datasets_and_model(
                model_class, dataset_conf, model_conf, train_conf, 
                is_train, False, False, True)

        for path in repl_utt_wspec.split(":")[1].split(","):
            check_and_makedirs(os.path.dirname(path))
        with BFMWriter(repl_utt_wspec) as writer:
            write_fn = lambda uttid, feat_3d: writer.write(uttid, flatten_channel(feat_3d))
            repl_repr_utt(
                    exp_dir, model, test_set, repr_set_name, 
                    repr_id_map, label_str_to_repr, repl_utt_list, write_fn)

        check_and_makedirs(repl_utt_img_dir)
        plot_kaldi_feat(repl_utt_wspec, repl_utt_img_dir, dataset_conf["feat_cfg"]["feat_type"])
        unset_logger()

def _load_configs(flags, model_parser, is_train):
    """use fixed #batches for each epoch"""
    # load and copy configurations
    exp_dir = flags.exp_dir
    check_and_makedirs(exp_dir)

    if is_train:
        maybe_copy(flags.model_conf, "%s/model.cfg" % exp_dir)
        maybe_copy(flags.train_conf, "%s/train.cfg" % exp_dir)
        maybe_copy(flags.dataset_conf, "%s/dataset.cfg" % exp_dir)

    model_conf = model_parser("%s/model.cfg" % exp_dir).get_config()
    train_conf = train_parser("%s/train.cfg" % exp_dir).get_config()
    dataset_conf = dataset_parser("%s/dataset.cfg" % exp_dir).get_config()
    if not is_train:
        train_conf["bs"] = 2048
    if not is_train and bool(flags.feat_rspec):
        dataset_conf["test_feat_rspec"] = flags.feat_rspec
        if bool(flags.feat_label_N) and bool(flags.feat_utt2label_path):
            dataset_conf["test_utt2label_paths"] = {}
            set_name_list = flags.feat_set_name.split(",")
            label_N_list = flags.feat_label_N.split(",")
            utt2label_path_list = flags.feat_utt2label_path.split(",")
            assert(len(label_N_list) == len(set_name_list))
            assert(len(label_N_list) == len(utt2label_path_list))
            for set_name, N, path, in \
                    zip(set_name_list, label_N_list, utt2label_path_list):
                dataset_conf["test_utt2label_paths"][set_name] = (int(N), path)
        else:
            dataset_conf["test_utt2label_paths"] = {}
        if bool(flags.feat_talabel_N) and bool(flags.feat_utt2talabels_path):
            dataset_conf["test_utt2talabels_paths"] = {}
            ta_set_name_list = flags.feat_ta_set_name.split(",")
            talabel_N_list = flags.feat_talabel_N.split(",")
            utt2talabels_path_list = flags.feat_utt2talabels_path.split(",")
            assert(len(talabel_N_list) == len(ta_set_name_list))
            assert(len(talabel_N_list) == len(utt2talabels_path_list))
            for set_name, N, path, in \
                    zip(ta_set_name_list, talabel_N_list, utt2talabels_path_list):
                dataset_conf["test_utt2talabels_paths"][set_name] = (int(N), path)
        else:
            dataset_conf["test_utt2talabels_paths"] = {}
        info("replaced test_feat_rspec with %s, utt2label_paths %s, utt2talabels_path %s" % (
            dataset_conf["test_feat_rspec"], dataset_conf["test_utt2label_paths"],
            dataset_conf["test_utt2talabels_paths"]))

    if model_conf["n_bins"] != dataset_conf["n_bins"]:
        raise ValueError("model and dataset n_bins not matched (%s != %s)" % (
                model_conf["n_bins"], dataset_conf["n_bins"]))

    return exp_dir, model_conf, train_conf, dataset_conf

def _print_configs(exp_dir, model_conf, train_conf, dataset_conf):
    info("Experiment Directory:\n\t%s" % str(exp_dir))
    info("Model Configurations:")
    for k, v in sorted(model_conf.items()):
        info("\t%s : %s" % (k.ljust(20), v))
    info("Training Configurations:")
    for k, v in sorted(train_conf.items()):
        info("\t%s : %s" % (k.ljust(20), v))
    info("Dataset Configurations:")
    for k, v in sorted(dataset_conf.items()):
        info("\t%s : %s" % (k.ljust(20), v))
    
def _load_datasets_and_model(
        VAE, dataset_conf, model_conf, train_conf,
        is_train, train, dev, test):
    # initialize dataset and model, create directories
    sets = datasets_loader(dataset_conf, train, dev, test)
    _set = [s for s in sets if s is not None][0]
    model_conf["input_shape"] = _set.feat_shape
    model_conf["target_shape"] = _set.feat_shape
    model_conf["target_dtype"] = tf.float32
    if model_conf["n_bins"] is not None:
        model_conf["target_dtype"] = tf.int32 
    model = VAE(model_conf, train_conf, training=is_train)
    return sets, model

def _load_dump_spk_repr_configs(flags, model_parser):
    assert(flags.spk_repr_id_map)
    assert(flags.spk_repr_spec)

    exp_dir, model_conf, train_conf, dataset_conf = \
            _load_configs(flags, model_parser, is_train=False)
    dataset_conf, _, _ = get_frame_ra_dataset_conf(dataset_conf)
    spk_repr_spec = flags.spk_repr_spec
    spk_repr_set_name = "spk"
    spk_repr_id_map = _load_id_map(flags.spk_repr_id_map)

    return exp_dir, model_conf, train_conf, dataset_conf, \
            spk_repr_set_name, spk_repr_id_map, spk_repr_spec

def _load_dump_phone_repr_configs(flags, model_parser):
    assert(flags.phone_repr_id_map)
    assert(flags.phone_repr_spec)

    exp_dir, model_conf, train_conf, dataset_conf = \
            _load_configs(flags, model_parser, is_train=False)
    dataset_conf, _, _ = get_frame_ra_dataset_conf(dataset_conf)
    phone_repr_spec = flags.phone_repr_spec
    phone_repr_set_name = "phone"
    phone_repr_id_map = _load_id_map(flags.phone_repr_id_map)

    return exp_dir, model_conf, train_conf, dataset_conf, \
            phone_repr_set_name, phone_repr_id_map, phone_repr_spec

def _load_repl_spk_repr_utt_configs(flags, model_parser):
    assert(flags.repl_utt_wspec)
    assert(flags.repl_utt_list)
    assert(flags.repl_utt_img_dir)

    exp_dir, model_conf, train_conf, dataset_conf, repr_set_name, \
            repr_id_map, repr_spec = _load_dump_spk_repr_configs(flags, model_parser)
    repl_utt_wspec = flags.repl_utt_wspec
    repl_utt_img_dir = flags.repl_utt_img_dir
    repl_utt_list = [line.rstrip().split() for line in open(flags.repl_utt_list)]
    label_str_to_repr = _load_repr(repr_spec)

    return exp_dir, model_conf, train_conf, dataset_conf, repr_set_name, \
            repr_spec, repr_id_map, label_str_to_repr, \
            repl_utt_list, repl_utt_wspec, repl_utt_img_dir

def _load_id_map(id_map_path):
    """id_map_path is the path to a file with "<key> <id>"" format
    output is a dictionary map from id (int) to key (str)"""
    id_map = {}
    with open(id_map_path) as f:
        id_content = f.readlines()
    for l in id_content:
        l = l.split(' ')
        id_map[int(l[1])] = l[0]
    return id_map

def _load_comp_ortho_configs(flags, model_parser):
    assert(flags.spk_repr_spec)
    assert(flags.phone_repr_spec)
    assert(flags.ortho_img_path)
    exp_dir, _, _, _ = _load_configs(flags, model_parser, is_train=False)
    spk_repr_spec = flags.spk_repr_spec
    phone_repr_spec = flags.phone_repr_spec
    ortho_img_path = flags.ortho_img_path
    return exp_dir, ortho_img_path, spk_repr_spec, phone_repr_spec

def _load_repr(repr_rspec):
    label_str_to_repr = dict()
    with SBFVReader(repr_rspec) as f:
        while not f.done():
            label_str, attr = f.next()
            label_str_to_repr[label_str] = attr
    return label_str_to_repr

def _comp_abs_cos(img_path, *repr_rspecs):
    print_labels = [
            "fjlm0", "felc0", "fmld0", "mwew0", "mjln0", "mjdh0", 
            "aa", "ae", "ay", "eh", "b", "d", "s", "sh", "m", "n"]
    label_str_to_repr_list = [_load_repr(repr_rspec) for repr_rspec in repr_rspecs]
    label_str_to_repr = dict([p for l2r in label_str_to_repr_list for p in l2r.items()])
    
    label_strs = [l for l in print_labels if l in label_str_to_repr.keys()]
    reprs = np.array([label_str_to_repr[l] for l in label_strs])

    norm_reprs = reprs / np.sqrt(np.sum(np.power(reprs, 2), axis=-1, keepdims=True))
    abs_cos_sim = np.abs(np.dot(norm_reprs, np.transpose(norm_reprs)))
    check_and_makedirs(os.path.dirname(img_path))
    plot_heatmap(abs_cos_sim, label_strs, label_strs, mode="save", name=img_path, figsize=(5, 3))
        
if __name__ == "__main__":
    main()
