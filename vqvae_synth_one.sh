#!/bin/bash -x
# Dec 2020
#
# Synthesise a single test sample using VQVAE

PATH=$PATH:~/codec2/build_linux/src

if [ "$#" -lt 2 ]; then
    echo "usage: ./vqvae_synth_one.sh sample npyfile out_dir"
    exit
fi

x=$(basename $1)
s="${x%.*}"
npy=$2
out_dir=$3
tmp_dir=$(mktemp -d)

c2sim $1 --rateK --rateKout ${tmp_dir}/${s}.f32
./vq_vae_kmeans_conv1d_out.py ${npy} ${tmp_dir}/${s}.f32 --featurefile_out ${tmp_dir}/${s}_out.f32 --eband_K 20 --scale 0.005 --mean
c2sim $1  --rateK --rateKin ${tmp_dir}/${s}_out.f32 -o ${out_dir}/${s}_out.raw
c2sim $1  --rateK --rateKin ${tmp_dir}/${s}_out.f32 --phase0 --postfilter -o ${out_dir}/${s}_outp0.raw
