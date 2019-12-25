#!/bin/bash -x
# David Dec 2019
#
# Synthesise test samples using AmpNN

PATH=$PATH:~/codec2/build_linux/src
if [ "$#" -lt 1 ]; then
    echo "usage: ./synth.sh out_dir nn.h5"
    exit
fi

SAMPLES="cap_8k experienced_8k fish_8k swam_8k"

out_dir=$1
nn=$2
mkdir -p $out_dir
tmp_dir=$(mktemp -d)
ls $tmp_dir
sox_args="-t .sw -r 8000 -c 1"

for s in $SAMPLES
do
    cp wav/$s.wav $out_dir
    bands=$tmp_dir/$s'_bands.f32'
    modelin=$tmp_dir/$s'.model'
    modelout=$tmp_dir/$s'_out.model'
    c2sim wav/$s.sw --bands $bands --modelout $modelin
    ./eband_out.py $nn $bands $modelin --modelout $modelout --noplots
    c2sim wav/$s.sw -o - | sox $sox_args - $out_dir/$s'_0_out.wav'
    c2sim wav/$s.sw -o - --phase0 --postfilter | sox $sox_args - $out_dir/$s'_1_p0.wav'
    c2sim wav/$s.sw --modelin $modelout -o - | sox $sox_args - $out_dir/$s'_2_nn.wav'
    c2sim wav/$s.sw --modelin $modelout -o - --phase0 --postfilter | sox $sox_args - $out_dir/$s'_3_nnp0.wav'
done

