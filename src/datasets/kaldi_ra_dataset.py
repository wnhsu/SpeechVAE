"""Kaldi Random Access Dataset Class"""
import os
import sys
import time
import numpy as np
import cPickle
from collections import defaultdict

from datasets import *
from datasets.base_dataset import BaseDataset
from datasets.label import load_time_aligned_labels, load_label
from tools.kaldi.prep_kaldi_feat import unflatten_channel
from kaldi_io import SequentialBaseFloatMatrixReader as SBFMReader
from kaldi_io import RandomAccessBaseFloatMatrixReader as RABFMReader

def _load_label_from_spec(label_spec, load_label, utt_list=None):
    """
    load label from specifier, which is a dict of label name to (#class, label path)
    if utt_list is provided, check if every utterance maps to one label
    """
    start_time = time.time()
    set_names = []
    n_class_list = []
    utt2label_list = []
    
    if label_spec:
        set_names, n_class_and_path_list = zip(*label_spec.items())
        for n_class, path in n_class_and_path_list:
            utt2label = load_label(path)
            if utt_list:
                _check_has_label(utt_list, utt2label)
            n_class_list.append(n_class)
            utt2label_list.append(utt2label)
        
    n_class_dict = IndexedDict(set_names, n_class_list)
    utt2label_dict = IndexedDict(set_names, utt2label_list)

    info("total %s label sets, keys are %s, n_class are %s, loading takes %.2f s" % (
        len(set_names), set_names, str(n_class_list), time.time() - start_time))
    return n_class_dict, utt2label_dict

def _check_has_label(utt_list, utt_to_label):
    for utt_id in utt_list:
        if not utt_id in utt_to_label:
            msg = "%s is not in utt_to_label:\n" % utt_id
            msg += "\tfirst few utt_to_label is %s" % str(utt_to_label.items()[:5])
            raise ValueError(msg)

def _make_seg_list(
        utt_index_list, utt_list, utt_len_list, seg_len, seg_shift, if_seg_rand, utt2label=None):
    """generate a list of (utt_id, start_f, end_f, None)
    INPUTS:
        if_seg_rand     - randomize segmentation within an utterance
    
    OUTPUTS:
        seg_list        - list of (utt_id, start_f, end_f). 
                          start_f is starting frame; end_f is ending frame
    """
    seg_list = []
    for utt_index in utt_index_list:
        utt_id = utt_list[utt_index]
        utt_len = utt_len_list[utt_index]
        label = utt2label[utt_id] if utt2label else None
        n_segs = (utt_len - seg_len) // seg_shift + 1
        if if_seg_rand:
            start_f_list = np.random.choice(xrange(utt_len - seg_len + 1), n_segs)
        else:
            start_f_list = np.arange(n_segs) * seg_shift
        for f in start_f_list:
            seg_list.append((utt_id, f, f + seg_len, label))
    debug(seg_list[:5])
    return seg_list

def _make_talabel_seg_list(
        utt_index_list, utt_list, utt_len_list, seg_len, utt2talabels):
    """generate a list of (utt_id, start_f, end_f) centered at time-aligned labels
    INPUTS:
    
    OUTPUTS:
        seg_list        - list of (utt_id, start_f, end_f, seg_label). 
                          start_f is starting frame; end_f is ending frame
    """
    seg_list = []
    for utt_index in utt_index_list:
        utt_id = utt_list[utt_index]
        utt_len = utt_len_list[utt_index]
        utt_segs = [talabel.get_centered_seg(seg_len, max_t=utt_len) \
                for talabel in utt2talabels[utt_id]]
        seg_list += [(utt_id, seg[0], seg[1], seg[2]) \
                for seg in utt_segs if seg is not None]
    debug(seg_list[:5])
    return seg_list

class KaldiRADataset(BaseDataset):
    """
    kaldi random access dataset class.

        1. more flexible memory-constrained data reader
        2. run time segment (sub-sequence) generation
    """
    def __init__(self, **kwargs):
        """
        Args:
            feat_rspec          - str of kaldi read specifier
            seg_len             - segment length
            seg_shift           - frame-shift between segments
            n_chan              - int of #channels for recovering the original
                                  shape (kaldi stores flattened feature for 
                                  each frame)
            use_chan            - list of int of which channels to use. 
                                  None for all
            use_fbin            - list of int of which frequency bins to use. 
                                  None for all
            seg_rand            - randomize segmentation of sequences if true
            if_rand             - bool of if shuffle the data plan
            mvn_path            - use meav variance params from this path, or 
                                  compute and save if file does not exist. None 
                                  for no mean variance normalization
            max_to_load         - int, max #utterances to load. -1 for loading all
            utt2label_paths     - dict of (n_class, label_path) tuple
            utt2talabels_paths  - dict of (n_class, time_aligned_labels_path) tuple
        """
        info("KaldiRADataset constructor")
        super(KaldiRADataset, self).__init__(**kwargs)

    def _set_conf(self,
                  feat_rspec,
                  seg_len,
                  seg_shift,
                  n_chan,
                  use_chan=None,
                  use_fbin=None,
                  seg_rand=False,
                  if_rand=False,
                  mvn_path=None,
                  max_to_load=-1,
                  utt2label_paths=None,
                  utt2talabels_paths=None,
                  **kwargs):
        debug("unused arguments: %s" % str(kwargs))
        self._feat_rspec        = feat_rspec
        self._seg_len           = seg_len
        self._seg_shift         = seg_shift
        self._n_chan            = n_chan
        self._use_chan          = use_chan
        self._use_fbin          = use_fbin
        self._seg_rand          = seg_rand
        self._if_rand           = if_rand
        self._mvn_path          = mvn_path
        self._max_to_load       = max_to_load
        self._utt2label_specs   = utt2label_paths
        self._utt2talabels_specs = utt2talabels_paths

    def _load_data(self):
        self._load_kaldi_feat_list()
        self._load_utt2label()
        self._load_utt2talabels()

    def _load_kaldi_feat_list(self):
        """
        generate a dict of random access table
        """
        start_time = time.time()
        self._utt_list = []
        self._utt_len_list = []
        self._utt2rawfeat = dict()
        with SBFMReader(self._feat_rspec) as f:
            while not f.done():
                utt_id, utt_feats = f.next()
                if len(utt_feats) < self._seg_len:
                    info("%s len (%s) shorter than seg_len (%s), discarded" % (
                            utt_id, len(utt_feats), self._seg_len))
                else:
                    self._utt_list.append(utt_id)
                    self._utt_len_list.append(len(utt_feats))
                if len(self._utt_list) % 500 == 0:
                    info("scanned %s utts" % len(self._utt_list))

        utt_feats = unflatten_channel(utt_feats, self._n_chan)
        utt_feats = utt_feats[self._use_chan, :, self._use_fbin]
        self._feat_shape = (utt_feats.shape[0], self._seg_len, utt_feats.shape[2])
        self._feat_dim = np.prod(self._feat_shape)

        self._kaldi_reader = RABFMReader(self._feat_rspec)
        info("scanning kaldi feat takes %.2f s, #utt=%s, #frames=%s, feat shape is %s, dim is %s" % (
                time.time() - start_time, len(self._utt_list), 
                sum(self._utt_len_list), self._feat_shape, self._feat_dim))

    def _load_utt2label(self):
        """
        load mapping of utt_id to label from file(s)
        """
        self._n_class_sets, self._utt2label_sets = \
                _load_label_from_spec(self._utt2label_specs, load_label, self._utt_list)

    def _load_utt2talabels(self):
        """
        load mapping of utt-id to time-aligned labels from file(s)
        """
        self._n_ta_class_sets, self._utt2talabels_sets = \
                _load_label_from_spec(self._utt2talabels_specs, load_time_aligned_labels, self._utt_list)

    def _init_data_plan(self):
        pass

    def _make_utt_plan(self, utt_iter_index, if_rand):
        if not isinstance(utt_iter_index, list):
            raise ValueError("utt_iter_index is not list (%s)" % (
                    type(utt_iter_index)))
        if if_rand:
            np.random.shuffle(utt_iter_index)
        else:
            utt_iter_index.sort()

    def _load_kaldi_feat(self, utt_index_list):
        """load raw features to memory
        if self._max_to_load, clean cached; else check and load
        """
        if self._max_to_load > 0:
            del self._utt2rawfeat
            self._utt2rawfeat = dict()
        # fast check if having loaded all utterances
        if len(self._utt2rawfeat) == len(self._utt_list):
            return
        for utt_index in utt_index_list:
            utt_id = self._utt_list[utt_index]
            # only load those which are not in memory
            if not utt_id in self._utt2rawfeat:
                feats = self._kaldi_reader[utt_id]
                self._utt2rawfeat[utt_id] = unflatten_channel(
                        feats, self._n_chan)

    def _make_seg_list(self, utt_index_list, if_seg_rand, if_rand, set_name=None):
        utt2label = None if set_name is None else self._utt2label_sets[set_name]
        seg_list = _make_seg_list(
                utt_index_list, self._utt_list, self._utt_len_list, 
                self._seg_len, self._seg_shift, if_seg_rand, utt2label)
        if if_rand:
            np.random.shuffle(seg_list)
        return seg_list

    def _make_talabel_seg_list(self, utt_index_list, if_rand, set_name):
        utt2talabels = self._utt2talabels_sets[set_name]
        seg_list = _make_talabel_seg_list(
                utt_index_list, self._utt_list, self._utt_len_list,
                self._seg_len, utt2talabels)
        if if_rand:
            np.random.shuffle(seg_list)
        return seg_list
    
    def _iterator(self, bs=256, set_name=None, utt_index_list=None, is_talabel=False):
        """iterator function
        by default iterate over all utterances""" 
        if utt_index_list is None:
            utt_index_list = range(len(self._utt_list))
        if is_talabel:
            assert(set_name in self._utt2talabels_sets)
        elif set_name is not None:
            assert(set_name in self._utt2label_sets)

        rem_batch = []
        n_utts = len(utt_index_list)
        self._make_utt_plan(utt_index_list, self._if_rand)
        next_utt_iter_pos = 0
        while next_utt_iter_pos != n_utts:
            start = next_utt_iter_pos if self._max_to_load > 0 else 0
            end = min(n_utts, start + self._max_to_load) if self._max_to_load > 0 else n_utts
            debug("next_utt_iter_pos = %s, max_to_load %s, #utt2rawfeat %s:" % (
                next_utt_iter_pos, self._max_to_load, len(self._utt2rawfeat)))
            
            start_time = time.time()
            self._load_kaldi_feat(utt_index_list[start:end])
            next_utt_iter_pos = end
            debug("...loading utt %s to %s takes %.2f s" % (start, end, time.time() - start_time) + \
                    ", loaded utt_id[:5]:%s" % str(self._utt2rawfeat.keys()[:5]))

            start_time = time.time()
            if is_talabel:
                seg_list = self._make_talabel_seg_list(
                        utt_index_list[start:end], self._if_rand, set_name)
            else:
                seg_list = self._make_seg_list(
                        utt_index_list[start:end], self._seg_rand, self._if_rand, set_name)
            n_segs, n_batches = len(seg_list), len(seg_list) // bs
            debug("...making seg list takes %.2f s" % (time.time() - start_time,) + \
                    ", #segs = %s, #batches = %s, seg_rand=%s" % (n_segs, n_batches, self._seg_rand))

            next_seg_pos = 0
            while next_seg_pos != n_segs:
                n_rem = len(rem_batch[0]) if bool(rem_batch) else 0
                start = next_seg_pos
                end = min(n_segs, start + bs - n_rem)
                batch = self._get_item(seg_list[start:end], rem_batch)
                next_seg_pos = end
                if len(batch[0]) == bs:
                    rem_batch = []
                    debug("yielded up to seg_pos = %s" % next_seg_pos)
                    yield batch
                else:
                    rem_batch = batch
                    debug("not yielding seg_pos = %s" % next_seg_pos + \
                            ", %s remained" % (len(rem_batch[0])))

        if bool(rem_batch):
            debug("yielding rem seg_pos = %s" % next_seg_pos)
            yield rem_batch

    def talabel_iterator(self, bs=256, set_name=None, utt_index_list=None):
        for batch in self._iterator(bs, set_name, utt_index_list, is_talabel=True):
            yield batch

    def iterator(self, bs=256, set_name=None, utt_index_list=None):
        for batch in self._iterator(bs, set_name, utt_index_list, is_talabel=False):
            yield batch

    def iterator_by_label(self, bs, set_name, label):
        label_to_utt_ids = self.get_label_utt_ids(set_name)
        utt_id_to_index = dict(zip(self._utt_list, range(len(self._utt_list))))
        utt_index_list = [
                utt_id_to_index[utt_id] for utt_id in label_to_utt_ids[label]]
        for batch in self.iterator(bs, set_name, utt_index_list):
            yield batch

    def _get_item(self, batch_segs, rem_batch=[]):
        """
        """
        use_c, use_f = self._use_chan, self._use_fbin
        
        labels = np.array([seg_label for _, _, _, seg_label in batch_segs])
        raw_feats = [self._utt2rawfeat[utt_id][use_c, start_f:end_f, use_f] \
                for utt_id, start_f, end_f, _ in batch_segs]
        feats = self.apply_mvn(np.asarray(raw_feats).astype(NP_FLOAT))
        masks = np.ones_like(feats).astype(NP_FLOAT)
        targets = np.array(feats)
        if bool(rem_batch):
            debug("rem_batch not empty, concat %s and %s" % (len(feats), len(rem_batch[0])))
            feats = np.concatenate([rem_batch[0], feats])
            masks = np.concatenate([rem_batch[1], masks])
            labels = np.concatenate([rem_batch[2], labels])
            targets = np.concatenate([rem_batch[3], targets])

        return feats, masks, labels, targets

    def _compute_mvn_and_save(self, path):
        n = 0.
        x = 0.
        x2 = 0.
        n_utts = 0
        with SBFMReader(self._feat_rspec) as f:
            while not f.done():
                _, utt_feats = f.next()
                utt_feats = unflatten_channel(utt_feats, self._n_chan)
                x += np.sum(utt_feats, axis=1, keepdims=True)
                x2 += np.sum(utt_feats ** 2, axis=1, keepdims=True)
                n += utt_feats.shape[1]
                n_utts += 1
                if n_utts % 500 == 0:
                    info("accumulated %s utts" % n_utts)
        mean = x / n
        std = np.sqrt(x2 / n - mean ** 2)
        info("mean shape is %s, value is\n%s" % (mean.shape, mean))
        info("std shape is %s, value is\n%s" % (std.shape, std))
        self._mvn_params = {"mean": mean, "std": std}
        check_and_makedirs(os.path.dirname(path))
        with open(path, "wb") as f:
            info("dumping mvn params to %s" % path)
            cPickle.dump(self._mvn_params, f)

    def apply_mvn(self, batch):
        assert(isinstance(batch, np.ndarray))
        if self._mvn_params is None:
            return batch
        else:
            use_c, use_f = self._use_chan, self._use_fbin
            mean = self._mvn_params["mean"][use_c, :, use_f]
            std = self._mvn_params["std"][use_c, :, use_f]
            return (batch - mean) / std

    def undo_mvn(self, batch):
        assert(isinstance(batch, np.ndarray))
        if self._mvn_params is None:
            return batch
        else:
            use_c, use_f = self._use_chan, self._use_fbin
            mean = self._mvn_params["mean"][use_c, :, use_f]
            std = self._mvn_params["std"][use_c, :, use_f]
            return batch * std + mean

    def feat_to_target(self, feats):
        return feats

    def target_to_feat(self, targets):
        return targets

    def get_label(self, set_name, utt_id):
        return self._utt2label_sets[set_name][utt_id]

    def get_label_N(self, set_name):
        label_to_N = defaultdict(int)
        for utt_id, utt_len in zip(self._utt_list, self._utt_len_list):
            label = self.get_label(set_name, utt_id)
            n_segs = (utt_len - self._seg_len) // self._seg_shift + 1
            label_to_N[label] += n_segs
        return label_to_N

    def get_label_utt_ids(self, set_name):
        label_to_utt_ids = defaultdict(list)
        for utt_id in self._utt_list:
            label = self.get_label(set_name, utt_id)
            label_to_utt_ids[label].append(utt_id)
        return label_to_utt_ids

    def get_n_class(self, set_name):
        return self._n_class_sets[set_name]

    @property
    def num_samples(self):
        raise NotImplementedError

    @property
    def feats(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

    @property
    def feat_shape(self):
        return self._feat_shape

    @property
    def feat_dim(self):
        return self._feat_dim
