# Amplitude Sample Rate Conversion Using Neural Nets

Codec 2 models speech as a harmonic series of sine waves, each with it's own frequency, amplitude and phase.  The frequencies are approximated at harmonics of the pitch or fundamental frequency.  A reasonable model of the phases can be recovered from the amplitudes.

Accurate representation of the sine wave amplitudes {Am} m=1...L is important for good quality speech.  The number of amplitudes in each frame L is dependent on the pitch L=P/2, which is time varying between (typically between L=10 and L=80).  However for transmission at a fixed bit rate, a fixed number of parameters is desirable.

In earlier Codec 2 modes such as 3200 down to 1200, the amplitudes were represented using a fixed number of Linear Prediction Coefficients.  In more recent modes such as 700C, the amplitudes Am are resampled to a fixed sample rate (K=20), and vector quantised for transmission.  At the decoder, the rate K amplitude samples are resampled back to rate L for synthesis.

This work explores the use of Neural Networks (NN) in resampling between rate K and rate L.  In [1] it was show that a set of quantised Codec 2 LPC parameters can be used to synthesise high quality speech, much higher than the corresponding Codec 2 decoder. In [2] it was shown that Mel spaced energy bands can be used to synthesise high quality speech.  Both suggest the amplitude information required to synthesise high quality speech can be contained in a small number of parameters, at distortion levels lower than traditional DSP.

Reducing the number of parameters K for a given quality level is desirable as it reduces the number of bits required to vector quantise at a given distortion.  Vector quantisers can efficiently encode correlated data, however after the initial correlation is removed the number of bits required for a further reduction in error is proportional to the dimension of the vector. Thus minimising the number of parameters is important for efficient low bit rate quantisation.

A key feature of deep NNs in this application is a non-linear mapping between the K parameters and the L amplitudes, unlike traditional linear resampling techniques.  This can overcome some resampling and quantisation distortion (e.g. undersampling of formants), and provide a useful tool for reducing the dimension of the vectors that require transmission over the channel.

# References

[1] [Wavenet based low rate speech coding](https://arxiv.org/abs/1712.01120)
[2] [LPCNet] (https://github.com/mozilla/LPCNet)

# Running the Simulations

Training:
```
$ ~/codec2/build_linux/src/c2sim ~/Downloads/all_speech_8k.sw --bands all_speech_8k.f32 --modelout all_speech_8k.model 
$ ./eband_train.py all_speech_8k.f32 all_speech_8k.model --epochs 25 --nnout ampnn.h5
```

Synthesise a single output sample:
```
$ ~/codec2/build_linux/src/c2sim wav/fish_8k.sw --bands fish_8k.f32 --modelout fish_8k.model 
$ ./eband_out.py ampnn.h5 fish_8k.f32 fish_8k.model
```

Synthesise a bunch of samples, place them in the directory called 191223:
```
$ ./synth.sh 191223.h5
```
