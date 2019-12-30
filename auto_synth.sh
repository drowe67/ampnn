#!/bin/bash -x

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc
tmp_dir="auto_dir"
mkdir -p $tmp_dir
#s=cap_8k
#s=fish_8k
s=experienced_8k
K=14
nn="ampnn.h5"
sox_args="-t .sw -r 8000 -c 1"

bands=$tmp_dir/$s'_bands.f32'
bands_q=$tmp_dir/$s'_bands_q.f32'
modelin=$tmp_dir/$s'.model'
modelout=$tmp_dir/$s'_out.model'
vec=$tmp_dir/vec.f32
vec_q=$tmp_dir/vec_q.f32

c2sim wav/$s.sw --bands $bands --modelout $modelin
./eband_auto.py $bands --dec 3 --nnin autonn.h5 --encout $vec --noplots
cat $vec | vq_mbest -k $K -q vq1.f32 -m 1 > $vec_q
./eband_auto.py $bands --dec 3 --nnin autonn.h5 --decin $vec_q --decout $bands_q --noplots

# passing thru ebands without any decimation or quatisation
./eband_out.py $nn $bands $modelin --modelout $modelout --eband_K $K --noplots
c2sim wav/$s.sw --modelin $modelout -o - --phase0 --postfilter | sox $sox_args - $s'_1.wav'

# autoencoder with VQ
./eband_out.py $nn $bands_q $modelin --modelout $modelout --eband_K $K --noplots
c2sim wav/$s.sw --modelin $modelout -o - --phase0 --postfilter | sox $sox_args - $s'_2.wav'

# autoencoder without VQ
./eband_auto.py $bands --dec 3 --nnin autonn.h5 --decin $vec --decout $bands_q --noplots
./eband_out.py $nn $bands_q $modelin --modelout $modelout --eband_K $K --noplots
c2sim wav/$s.sw --modelin $modelout -o - --phase0 --postfilter | sox $sox_args - $s'_3.wav'
