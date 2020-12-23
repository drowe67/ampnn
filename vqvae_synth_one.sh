#!/bin/bash -x
# Dec 2020
#
# Synthesise a single test sample using VQVAE

PATH=$PATH:~/codec2/build_linux/src

if [ "$#" -lt 2 ]; then
    echo "usage: ./vqvae_synth_one.sh sample out_dir"
    exit
fi

x=$(basename $1)
s="${x%.*}"
out_dir=$2
tmp_dir=$(mktemp -d)

c2sim $1 --rateK --rateKout ${tmp_dir}/${s}.f32
./vq_vae_kmeans_conv1d_out.py test.npy ${tmp_dir}/${s}.f32 ${tmp_dir}/${s}_out.f32 --eband_K 20 --scale 0.005 --mean
c2sim $1 --rateK --rateKin ${tmp_dir}/${s}_out.f32 -o ${out_dir}/${s}_out.raw
c2enc 700C $1 - | c2dec 700C - ${out_dir}/${s}_700c.raw
