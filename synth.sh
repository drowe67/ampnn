#!/bin/bash -x
# David Dec 2019
#
# Synthesise test samples using AmpNN

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc
if [ "$#" -lt 2 ]; then
    echo "usage: ./synth.sh out_dir nn.h5 [K_st K_en] [vq1] [vq2]"
    exit
fi

SAMPLES="cap_8k experienced_8k fish_8k swam_8k"
maxK=14
gain=10

out_dir=$1
nn=$2

if [ "$#" -ge 4 ]; then
    K_st=$3
    K_en=$4
else
    K_st=0
    K_en=13
fi
K=$(python -c "print(int(${K_en[i]}-${K_st[i]}+1))")

mkdir -p $out_dir
tmp_dir=$(mktemp -d)
ls $tmp_dir
sox_args="-t .sw -r 8000 -c 1"

for s in $SAMPLES
do
    cp wav/$s.wav $out_dir
    bands=$tmp_dir/$s'_bands.f32'
    bands_slice=$tmp_dir/$s'_bands_slice.f32'
    bands_quantised=$tmp_dir/$s'_bands_quantised.f32'
    modelin=$tmp_dir/$s'.model'
    modelout=$tmp_dir/$s'_out.model'
    modelout_quantised=$tmp_dir/$s'_out_quantised.model'
    c2sim wav/$s.sw --bands $bands --modelout $modelin
    extract -s $K_st -e $K_en -t $maxK $bands $bands_slice -g $gain

    ./eband_out.py $nn $bands_slice $modelin --modelout $modelout --eband_K $K --noplots --gain $gain
       
    c2sim wav/$s.sw -o - | sox $sox_args - $out_dir/$s'_0_out.wav'
    c2sim wav/$s.sw -o - --phase0 --postfilter | sox $sox_args - $out_dir/$s'_1_p0.wav'
    c2sim wav/$s.sw --modelin $modelout -o - | sox $sox_args - $out_dir/$s'_2_nn.wav'
    c2sim wav/$s.sw --modelin $modelout -o - --phase0 --postfilter | sox $sox_args - $out_dir/$s'_3_nnp0.wav'

    # optional one stage VQ
    if [ "$#" -eq 5 ]; then
	cat $bands_slice | vq_mbest -k $K -q $5 -m 1 > $bands_quantised
    fi
    # optional two stage VQ
    if [ "$#" -eq 6 ]; then
	cat $bands_slice | vq_mbest -k $K -q $5,$6 -m 4 > $bands_quantised
    fi
    
    if [ "$#" -ge 5 ]; then
	./eband_out.py $nn $bands_quantised $modelin --modelout $modelout_quantised --eband_K $K --noplots --gain $gain
	c2sim wav/$s.sw --modelin $modelout_quantised -o - --phase0 --postfilter | sox $sox_args - $out_dir/$s'_4_nnqp0.wav'
    fi
done

