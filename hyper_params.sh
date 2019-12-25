#!/bin/bash -x
# David Dec 2019
#

# Run VQ train and NN train over some broad "hyper" parameters, such
# as K and #VQ entries.  Note we're not actually messing with NN
# hyper-parms (yet).

PATH=$PATH:~/codec2/build_linux/src:$PATH:~/codec2/build_linux/misc


if [ "$#" -lt 2 ]; then
    echo "usage: ./hyper_params.sh train_prefix results_dir"
    echo "       ./hyper_params.sh all_speech_8k 191215"
    exit 0
else
    f=$1
    res_dir=$2
fi

maxK=14            # c2sim extracts vectors this wide
gain=10            # gives us dB from log10(band energys)
epochs=25
vq_stop=1E-3       # VQ training stop criterion

N=7                                      # number of trials

K_st=(  0    0    0    0    2    2    2) # start of slice
K_en=( 13   13   13   13   13   13   13) # end of slice
M1=(  512 1024 2048  512  512  512 2048) # number of VQ levels (same each stage)
M2=(    0    0    0  512  512  512    0) # number of VQ levels (same each stage)

echo ${#K_st[@]}
if [ ${#K_st[@]} -lt $N ]; then echo "K_st wrong length"; exit 1; fi
if [ ${#K_en[@]} -lt $N ]; then echo "K_en wrong length"; exit 1; fi
if [ ${#M1[@]} -lt $N ]; then echo "M1 wrong length"; exit 1; fi
if [ ${#M2[@]} -lt $N ]; then echo "M2 wrong length"; exit 1; fi

mkdir -p $res_dir
results=$res_dir/results.txt
printf "i\tK\tK_st\tK_en\tM1\tM2\tNNuq\tVQ\tNNvq1\tNNvq2\n" > $results

N_1=$(python -c "print(int($N-1))")
for i in $( seq 0 $N_1 )
do
    printf "================= Starting Iteration %d ===============\n" $i

    K=$(python -c "print(int(${K_en[i]}-${K_st[i]}+1))")
    printf "%d\t%d\t%d\t%d\t%d\t%d\t" $i $K ${K_st[i]} ${K_en[i]} ${M1[i]} ${M2[i]} >> $results

    # bunch of intermediate files
    train0=$res_dir/$i_'train0.f32'            # input rate K to VQ/NN training
    train1=$res_dir/$i_'train1.f32'            # stage1 VQ residual/stage 2 VQ input
    train2=$res_dir/$i_'train2.f32'
    quantised=$res_dir/$id_'quantised.f32'     # VQ quantised version of train0
    tmp=$res_dir/$i'_tmp.txt'
    vq1=$res_dir/$i'_vq1.f32'                  # stage1 VQ table
    vq2=$res_dir/$i'_vq2.f32'                  # stage2 VQ table
    nn=$res_dir/$i'_nn.h5'                     # eband NN trained from $train0
    nnvq=$res_dir/$i'_nnvq.h5'                 # eband NN trained from $quantised
    
    extract -s ${K_st[i]} -e ${K_en[i]} -t $maxK $f'.f32' $train0 -g $gain

    # Train NN on unquantised ebands as control
    ./eband_train.py $train0 $f'.model' --eband_K $K --epochs $epochs --nnout $nn --noplots --gain $gain > $tmp
    printf "%4.2f\t" `tail -n1 $tmp` >> $results
        
    # train a VQ
    extract -s ${K_st[i]} -e ${K_en[i]} -t $maxK $f'.f32' $train0 -g $gain
    vqtrain -s $vq_stop -r $train1 $train0 $K ${M1[i]} $vq1
    if  [ "${M2[i]}" -ne 0 ]; then
	vqtrain -s $vq_stop -r $train2 $train1 $K ${M2[i]} $vq2
    fi
    
    # VQ our training database to get quantised vectors
    if  [ "${M2[i]}" -ne 0 ]; then
	cat $train0 | vq_mbest -k $K -q $vq1,$vq2 -m 4 > $quantised 2>$tmp
    else
	cat $train0 | vq_mbest -k $K -q $vq1 -m 1 > $quantised 2>$tmp
    fi
    printf "%4.2f\t" `tail -n1 $tmp` >> $results
    
    # Evaulate previous NN on quantised output
    ./eband_out.py $nn $quantised $f'.model' --eband_K $K --noplots --gain $gain > $tmp
    printf "%4.2f\t" `tail -n1 $tmp` >> $results
    
    # Train new NN on quantised output of VQ
    ./eband_train.py $quantised $f'.model' --eband_K $K --epochs $epochs --noplots --nnout $nnvq --gain $gain > $tmp
    printf "%4.2f\n" `tail -n1 $tmp` >> $results
    
    # Synthesise output
    if  [ "${M2[i]}" -eq 0 ]; then
	./synth.sh $res_dir/$i'_wav' $nnvq ${K_st[i]} ${K_en[i]} $vq1
    else
	./synth.sh $res_dir/$i'_wav' $nnvq ${K_st[i]} ${K_en[i]} $vq1 $vq2
    fi
done

# TODO plot PNG with hyper param results

