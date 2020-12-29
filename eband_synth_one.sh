#!/bin/bash -x
# Dec 2020
#
# Synthesise a single test sample using eband_out rate K -> rate L

PATH=$PATH:~/codec2/build_linux/src

if [ "$#" -lt 2 ]; then
    echo "usage: ./eband_synth_one.sh sample h5File out_dir"
    exit
fi

x=$(basename $1)
s="${x%.*}"
ampnn=$2
out_dir=$3
tmp_dir=$(mktemp -d)

c2sim $1 --bands ${tmp_dir}/${s}.f32 --modelout ${tmp_dir}/${s}.model
./eband_out.py ${ampnn} ${tmp_dir}/${s}.f32 ${tmp_dir}/${s}.model --modelout ${tmp_dir}/${s}_out.model --noplots
c2sim $1 --modelin ${tmp_dir}/${s}_out.model -o ${out_dir}/${s}_out.raw
