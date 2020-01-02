#!/bin/bash -x
# David Dec 2019
#

# Run VQ train and NN train over some broad "hyper" parameters, such
# as K and #VQ entries.  Note we're not actually messing with NN
# hyper-parms (yet).

PATH=$PATH:~/codec2/build_linux/src:$PATH:~/codec2/build_linux/misc

if [ "$#" -lt 2 ]; then
    echo "usage: ./hyper_params.sh SpeechFile results_dir"
    echo "       ./hyper_params.sh ~/Downloads/all_8k.sw 191215"
    exit 0
else
    x=$(basename $1)
    f="${x%.*}"
    res_dir=$2
fi

maxK=14            # c2sim extracts vectors this wide
gain=10            # gives us dB from log10(band energys)
epochs=25
vq_stop=1E-3       # VQ training stop criterion

N=4                # number of trials

K_st=(  1    1     1     1) # start of slice
K_en=( 13   13    13    13) # end of slice
M1=(  512  512  4096  4096) # number of VQ levels (same each stage)
M2=(  512  512     0     0) # number of VQ levels (same each stage)
L=(     1    1     1     1) # lower log10(e) limit of samples to use for NN/VQ training
D=(     1    3     1     3) # frame rate decimation

if [ ${#K_st[@]} -lt $N ]; then echo "K_st wrong length"; exit 1; fi
if [ ${#K_en[@]} -lt $N ]; then echo "K_en wrong length"; exit 1; fi
if [ ${#M1[@]} -lt $N ]; then echo "M1 wrong length"; exit 1; fi
if [ ${#M2[@]} -lt $N ]; then echo "M2 wrong length"; exit 1; fi
if [ ${#L[@]} -lt $N ]; then echo "L wrong length"; exit 1; fi
if [ ${#D[@]} -lt $N ]; then echo "D wrong length"; exit 1; fi

mkdir -p $res_dir
results=$res_dir/results.txt
printf "i\tK\tK_st\tK_en\tM1\tM2\tDec\tNNuq\tVQ\tNNvq\n" > $results

N_1=$(python -c "print(int($N-1))")
for i in $( seq 0 $N_1 )
do
    printf "================= Starting Iteration %d ===============\n" $i

    K=$(python -c "print(int(${K_en[i]}-${K_st[i]}+1))")
    printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t" $i $K ${K_st[i]} ${K_en[i]} ${M1[i]} ${M2[i]} ${D[i]}  >> $results

    # bunch of intermediate files
    train0=$res_dir/$i'_train0.f32'            # input rate K to VQ and NN training (low energy vectors removed)
    train1=$res_dir/$i'_train1.f32'            # stage1 VQ residual/stage 2 VQ input
    train2=$res_dir/$i'_train2.f32'
    quantised=$res_dir/$id_'quantised.f32'     # VQ quantised version of train0
    tmp=$res_dir/$i'_tmp.txt'
    vq1=$res_dir/$i'_vq1.f32'                  # stage1 VQ table
    vq2=$res_dir/$i'_vq2.f32'                  # stage2 VQ table
    nn=$res_dir/$i'_nn.h5'                     # eband NN trained from $bands_slice
    
    # Train NN on unquantised ebands
    c2sim $1 --bands $f'.f32' --modelout $f'.model' --bands_lower ${L[i]}
    extract -s ${K_st[i]} -e ${K_en[i]} -t $maxK $f'.f32' $train0 -g $gain
    ./eband_train.py $train0 $f'.model' --eband_K $K --epochs $epochs --nnout $nn --noplots --gain $gain  > $tmp
    printf "%4.2f\t" `tail -n1 $tmp` >> $results
     
    # train a VQ
    vqtrain -s $vq_stop -r $train1 $train0 $K ${M1[i]} $vq1
    if  [ "${M2[i]}" -ne 0 ]; then
	vqtrain -s $vq_stop -r $train2 $train1 $K ${M2[i]} $vq2
    fi
    
    # VQ our training database to get quantised vectors and measure VQ error
    lower=$(python -c "print(int(${L[i]}*$gain))")
    if  [ "${M2[i]}" -ne 0 ]; then
	cat $train0 | vq_mbest -k $K -q $vq1,$vq2 -m 4 --lower $lower > $quantised 2>$tmp
    else
	cat $train0 | vq_mbest -k $K -q $vq1 -m 1 --lower $lower > $quantised --lower $lower 2>$tmp
    fi
    printf "%4.2f\t" `tail -n1 $tmp` >> $results
    
    # Run NN on quantised output (not sure if dec is a good idea on energy filtered vectors)
    ./eband_out.py $nn $quantised $f'.model' --eband_K $K --noplots --gain $gain --dec ${D[i]} > $tmp
    printf "%4.2f\n" `tail -n1 $tmp` >> $results
       
    # Synthesise output
    if  [ "${M2[i]}" -eq 0 ]; then
	LOWER=$lower DEC="${D[i]}" ./synth.sh $res_dir/$i'_wav' $nn ${K_st[i]} ${K_en[i]} $vq1
    else
	LOWER=$lower DEC="${D[i]}" ./synth.sh $res_dir/$i'_wav' $nn ${K_st[i]} ${K_en[i]} $vq1 $vq2
    fi
done

# TODO plot PNG with hyper param results

