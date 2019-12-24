#!/bin/bash -x
# David Dec 2019
#

# Run VQ train and NN train over some broad "hyper" parameters, such
# as K and #VQ entries.  Note we're not actually messing with NN
# hyper-parms (yet).

# TODO
#   [ ] log file with K, start/end, VQ rate K var, NN rate L var
#   [ ] controls at start without any VQ
#   [ ] ouput NN file, we can then use synth.sh to listen

PATH=$PATH:~/codec2/build_linux/src:$PATH:~/codec2/build_linux/misc

results="results.txt"

#f=all_speech_8k
f=all_8k

maxK=14       # c2sim extracts vectors this wide
gain=10       # gives us dB from log10(band energys)
epochs=1
K_st=(0  1)   # start of slice
K_en=(13 12)  # end of slice
M=(1024 1024) # number of VQ levels (same each stage)

tmp_dir=$(mktemp -d)
train0=$tmp_dir/train0.f32
train1=$tmp_dir/train1.f32
train2=$tmp_dir/train2.f32
quantised=$tmp_dir/quantised.f32
tmp=$tmp_dir/tmp.txt
vq1=$tmp_dir/vq1.f32
vq2=$tmp_dir/vq2.f32

printf "i\tK\tK_st\tK_en\tNNuq\tVQ\tNNvq\n" > $results

for i in {0..1}
do
    K=$(python -c "print(int(${K_en[i]}-${K_st[i]}+1))")
    printf "%d\t%d\t%d\t%d\t" $i $K ${K_st[i]} ${K_en[i]} >> $results
    
    extract -s ${K_st[i]} -e ${K_en[i]} -t $maxK $f'.f32' $train0 -g $gain

    # Train NN on unquantised ebands as control
    ./eband_train.py $train0 $f'.model' --eband_K $K --epochs $epochs --noplots --gain $gain > $tmp
    printf "%4.2f\t" `tail -n1 $tmp` >> $results
    
    # train a VQ
    extract -s ${K_st[i]} -e ${K_en[i]} -t $maxK $f'.f32' $train0 -g $gain
    vqtrain -r $train1 $train0 $K ${M[i]} $vq1
    vqtrain -r $train2 $train1 $K ${M[i]} $vq2
        
    # VQ our training datatbase to get quantised vectors
    cat $train0 | vq_mbest -k $K -q $vq1,$vq2 -m 4 > $quantised 2>$tmp
    printf "%4.2f\t" `tail -n1 $tmp` >> $results
    
    # Train NN on quantised output of this VQ
    ./eband_train.py $quantised $f'.model' --eband_K $K --epochs $epochs --noplots --gain $gain > $tmp
    printf "%4.2f\n" `tail -n1 $tmp` >> $results
done

# TODO plot PNG with hyper param results

