#!/bin/bash -x
# Dec 2020
#
# Synthesise a single test sample using VQVAE VQ, then eband_out rate K -> rate L

PATH=$PATH:~/codec2/build_linux/src

if [ "$#" -lt 2 ]; then
    echo "usage: ./_vqvae_eband_synth_one.sh sample vqvaeNN.npy ebandNN.h5 out_dir"
    exit
fi

x=$(basename $1)
s="${x%.*}"
vqvaenn=$2
ebandnn=$3
out_dir=$4
tmp_dir=$(mktemp -d)

in_f32=${tmp_dir}/${s}.f32
in_model=${tmp_dir}/${s}.model
out_f32=${tmp_dir}/${s}_out.f32
out_model=${tmp_dir}/${s}_out.model

c2sim $1 --bands $in_f32 --modelout $in_model

./vq_vae_kmeans_conv1d_out.py $vqvaenn $in_f32 --scale 0.005 --featurefile_out $out_f32
./eband_out.py $ebandnn $out_f32 $in_model --modelout $out_model --noplots
c2sim $1 --modelin $out_model -o ${out_dir}/${s}_out.raw
