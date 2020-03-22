This started out as Amplitude Sample Rate Conversion as described below, but is currently being used as a repo for my learning about various NN topics such as conv1d, variational autoencoders, and VQ-VAE. 

# Amplitude Sample Rate Conversion Using Neural Nets

Codec 2 models speech as a harmonic series of sine waves, each with it's own frequency, amplitude and phase.  The frequencies are approximated at harmonics of the pitch or fundamental frequency.  A reasonable model of the phases can be recovered from the amplitudes.

Accurate representation of the sine wave amplitudes {Am} m=1...L is important for good quality speech.  The number of amplitudes in each frame L is dependent on the pitch L=P/2, which is time varying between (typically between L=10 and L=80).  However for transmission at a fixed bit rate, a fixed number of parameters is desirable.

In earlier Codec 2 modes such as 3200 down to 1200, the amplitudes were represented using a fixed number of Linear Prediction Coefficients.  In more recent modes such as 700C, the amplitudes Am are resampled to a fixed sample rate (K=20), and vector quantised for transmission.  At the decoder, the rate K amplitude samples are resampled back to rate L for synthesis.

This work explores the use of Neural Networks (NN) in resampling between rate K and rate L.  In [1] it was show that a set of quantised Codec 2 LPC parameters can be used to synthesise high quality speech, much higher than the corresponding Codec 2 decoder. In [2] it was shown that Mel spaced energy bands can be used to synthesise high quality speech.  Both suggest the amplitude information required to synthesise high quality speech can be contained in a small number of parameters, at distortion levels lower than traditional DSP.

Reducing the number of parameters K for a given quality level is desirable as it reduces the number of bits required to vector quantise at a given distortion.  Vector quantisers can efficiently encode correlated data, however after the initial correlation is removed the number of bits required for a further reduction in error is proportional to the dimension of the vector. Thus minimising the number of parameters is important for efficient low bit rate quantisation.

A key feature of deep NNs in this application is a non-linear mapping between the K parameters and the L amplitudes, unlike traditional linear resampling techniques.  This can overcome some resampling and quantisation distortion (e.g. undersampling of formants), and provide a useful tool for reducing the dimension of the vectors that require transmission over the channel.

# Key Points

Novel things I developed/learned.  Not claiming they are original - just new to me:

* I'm using regression - the NN estimates the actual log10(Am) values, not discrete PDFs.
* Making NN works with variable L Am vectors, using a sparse target and custom loss functions.  Developed this during PhaseNN project (TODO link).
* Removed mean of rateK vectors before feeding to NN.  NNs seem to like having means removed.  This was worth several dB.
* The whole system (NN training and vector quantisation) works on mean square error in the log(Am) domain, which is equivalence to (i) variance (ii) and proportional to Spectral Distortion (SD) in dB^2 - which is very closely correlated to subjective quality.  Having an objective measure for developing a speech coding system is gold.  It trains in dB.
* I'm starting with flat filtered samples from a reputable database rather than my usual rag tag samples from random microphones and sound cards.  Removes one set of variables.  An equaliser (TODO REF) can make it usable for other input filtering.
* Scripts (TODO LINK) to automate cycling through various hyper parameter permutations and present nice little tables.
* Once again I have found low pitched samples (TODO) link evalution, hts1a difficult to code.  They have narrow formants, which gives them the dispersion necessary when harmonics are closely spaced.  Broaden the
formant and they get buzzy and muffled (lower intelligability)
* Low pitched speakers also caused problesm with harmonics close to Fs/2 (e.g. 3600-3800 Hz) which the anti-aliasing filter had greatly attenuated.  This caused a disproportional contribution to MSE.

# References

[1] [Wavenet based low rate speech coding](https://arxiv.org/abs/1712.01120)
[2] [LPCNet] (https://github.com/mozilla/LPCNet)

# Running the Simulations

Training:
```
$ ~/codec2/build_linux/src/c2sim ~/Downloads/all_speech_8k.sw --bands all_speech_8k.f32 --modelout all_speech_8k.model 
$ ./eband_train.py all_speech_8k.f32 all_speech_8k.model --epochs 25 --removemean --nnout ampnn.h5
```

Synthesise a single output sample:
```
$ ~/codec2/build_linux/src/c2sim wav/fish_8k.sw --bands fish_8k.f32 --modelout fish_8k.model 
$ ./eband_out.py ampnn.h5 fish_8k.f32 fish_8k.model --removemean --modelout fisk_8k_out.model
$ ~/codec2/build_linux/src/c2sim wav/fish_8k.sw --modelin fisk_8k_out.model -o - | aplay -f S16_LE
```

Synthesise a bunch of samples, place them in the directory called 191223:
```
$ ./synth.sh 191223.h5
```
