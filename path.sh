. /data/sls/scratch/wnhsu/env/cuda.sh
. /usr/users/wnhsu/vtenvs/tf1.0/bin/activate 

export PYTHONPATH=/data/sls/scratch/wnhsu/tmp/SpeechVAE/src:$PYTHONPATH
export PYTHONPATH=/data/sls/scratch/wnhsu/tmp/SpeechVAE/kaldi-python:$PYTHONPATH

export KALDI_ROOT=/data/sls/scratch/wnhsu/tmp/SpeechVAE/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$KALDI_ROOT/tools/openfst/bin:$PATH
export PATH=$KALDI_ROOT/egs/wsj/s5/utils:$PATH
export PATH=$KALDI_ROOT/egs/wsj:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present"
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
