#!/bin/bash -x
# Dec 2020
#
# Synthesise a single test sample using eband_vq_am rate K -> VQVAE -> rate L

PATH=$PATH:~/codec2/build_linux/src

if [ "$#" -lt 2 ]; then
    echo "usage: ./eband_va_am_synth_one.sh sample npyFile out_dir"
    exit
fi

x=$(basename $1)
s="${x%.*}"
nn_npy=$2
out_dir=$3
tmp_dir=$(mktemp -d)

in_f32=${tmp_dir}/${s}.f32
in_model=${tmp_dir}/${s}.model
out_f32=${tmp_dir}/${s}_out.f32
out_model=${tmp_dir}/${s}_out.model

c2sim $1 --bands $in_f32 --modelout $in_model

./eband_vq_am_out.py $nn_npy $in_f32 $in_model --modelout $out_model --nb_embedding 2048 --mean --scale 0.02 
c2sim $1 --modelin $out_model -o ${out_dir}/${s}_out.raw
