#!/bin/bash
./eband_auto.py all_speech_8k.f32 --epochs 25 --dec 3 --encout vec.f32 --noplots --overlap
~/codec2/build_linux/misc/vqtrain vec.f32 14 4096 vq1.f32
cat vec.f32 | ~/codec2/build_linux/misc/vq_mbest -k 14 -q vq1.f32 -m 1 > vec_q.f32
./eband_auto.py all_speech_8k.f32 --dec 3 --nnin autonn.h5 --decin vec_q.f32 --noplots
