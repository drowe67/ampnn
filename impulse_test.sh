#!/bin/bash -x
# Dec 2020
#
# Use synthetic signals to test eband_vq_am.py

PATH=$PATH:~/codec2/build_linux/src:~/codec2/build_linux/misc

timpulse --randf0 --secs 1200 --rande 4 --filter > impulse.raw
c2sim impulse.raw --bands impulse.f32 --modelout impulse.model
./model_to_sparse.py impulse.model impulse_sparse.f32
./eband_vq_am.py impulse.f32 impulse_sparse.f32 --nb_embedding 8 --epochs 5
