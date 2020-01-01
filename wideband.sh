#!/bin/bash -x
# script to run some wideband experiments

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc
K_st=0
K_en=13
maxK=18
sox_args="-t .sw -r 16000 -c 1"
tmp_dir=$(mktemp -d)
nn=ampwide.h5

function train() {
    s=all_speech
    bands=$s'.f32'
    bands_slice=$s'_slice.f32'
    model=$s'.model'
    cat ~/Downloads/$s'.sw' | c2sim - --bands $bands --modelout $model --Fs 16000
    extract -s $K_st -e $K_en -t $maxK $bands $bands_slice
    ./eband_train.py $bands_slice $model --Fs 16000 --nb_samples 100000 --epochs 25 --nnout $nn --noplots
    ./eband_out.py $nn $bands_slice $model --Fs 16000 --nb_samples 600 --modelout test_out.model
    c2sim ~/Downloads/$s'.sw' --modelin $model --Fs 16000 -o - | sox $sox_args - test_out.wav   
    c2sim ~/Downloads/$s'.sw' --modelin test_out.model --Fs 16000 -o - | sox $sox_args - test_nn.wav   
}

function synth() {
    bands=$tmp_dir/$1'.f32'
    bands_slice=$tmp_dir/$1'_slice.f32'
    model=$tmp_dir/$1'.model'
    modelout=$tmp_dir/$1'_out.model'
    c2sim wav/$1'.sw' --bands $bands --modelout $model --Fs 16000
    extract -s $K_st -e $K_en -t $maxK $bands $bands_slice
    ./eband_out.py $nn $bands_slice $model --modelout $modelout --Fs 16000
    c2sim wav/$1'.sw' --Fs 16000 -o - | sox $sox_args - $1'_out.wav'
    c2sim wav/$1'.sw' --modelin $modelout --Fs 16000 -o - | sox $sox_args - $1'_nn.wav'
}

#train
synth box
synth four

