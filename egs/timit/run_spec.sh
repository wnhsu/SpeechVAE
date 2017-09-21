#!/bin/bash 

# Copyright 2017 Wei-Ning Hsu
# Apache 2.0 

TIMIT_DIR=../../kaldi/egs/timit/s5
TIMIT_RAW_DATA=/usr/users/dharwath/data/timit

. path.sh || exit 1;
. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    echo "Usage: $0: [options]"
    exit 1;
fi

./local/spec_data_prep.sh --TIMIT_RAW_DATA $TIMIT_RAW_DATA \
    --TIMIT_KALDI_EGS $TIMIT_DIR --KALDI_ROOT $KALDI_DIR || exit 1;

# i/o configurations
exp_dir=exp/conv1_bn_spec_20_20
feat_dir=data/spec_scp
tt=test

tt_feat_rspec=scp:$feat_dir/$tt/feats.scp
tt_utt2uttid=$feat_dir/$tt/utt2uttid
tt_utt2spkid=$feat_dir/$tt/utt2spkid
tt_utt2phoneid=$feat_dir/$tt/utt2phoneid.talabel
tt_spk2spkid=$feat_dir/$tt/spk2spkid
tt_phone2phoneid=$feat_dir/$tt/phone2phoneid

tt_nutt=$(($(wc -l $tt_utt2uttid | awk '{print $1}') + 1))
tt_nspk=$(($(wc -l $tt_spk2spkid | awk '{print $1}') + 1))
tt_nphone=$(($(wc -l $tt_phone2phoneid | awk '{print $1}') + 1))

tt_repr_spk_wspec=ark:$exp_dir/eval/repr_spk/${tt}.ark
tt_repr_phone_wspec=ark:$exp_dir/eval/repr_phone/${tt}.ark
tt_ortho_img_path=$exp_dir/eval/ortho/abs_cos.png
tt_repl_spk_label=qual_conf/repl_spk.txt
tt_repl_utt_spk_wspec=ark:$exp_dir/eval/repl_utt_spk/${tt}.ark
tt_repl_utt_spk_img_dir=$exp_dir/eval/repl_utt_spk/img

# train model and run analysis
python src/scripts/run_is17_exp.py \
    --exp_dir=$exp_dir \
    --train_conf=conf/train/vae/train.cfg \
    --model_conf=conf/model/vae/spec_conv1_bn.cfg \
    --dataset_conf=data/spec_scp/train/dataset.cfg \
    --feat_rspec=$tt_feat_rspec \
    --feat_set_name=uttid,spk \
    --feat_label_N=$tt_nutt,$tt_nspk \
    --feat_utt2label_path=$tt_utt2uttid,$tt_utt2spkid \
    --feat_ta_set_name=phone \
    --feat_talabel_N=$tt_nphone \
    --feat_utt2talabels_path=$tt_utt2phoneid \
    --test \
    --dump_spk_repr \
    --spk_repr_id_map=$tt_spk2spkid \
    --spk_repr_spec=$tt_repr_spk_wspec \
    --dump_phone_repr \
    --phone_repr_id_map=$tt_phone2phoneid \
    --phone_repr_spec=$tt_repr_phone_wspec \
    --comp_ortho \
    --ortho_img_path=$tt_ortho_img_path \
    --repl_repr_utt \
    --repl_utt_list=$tt_repl_spk_label \
    --repl_utt_wspec=$tt_repl_utt_spk_wspec \
    --repl_utt_img_dir=$tt_repl_utt_spk_img_dir || exit 1;
