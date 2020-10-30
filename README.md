# Some experiments with Spectral Amplitude Quantisation using NNs

## Themes and Key Points

* I'm using Keras 2.3, Tensowflow 2.0
* For convenience I use Codec 2 (an old school, non NN vocoder) to "listen" to results from this work, however the cool kids are using VQ VAE with NN vocoders.
* I'm using regression - the NN estimates the actual log10(Am) values, not discrete PDFs.
* Making NN works with sparse (variable L) spectral magnitude vectors, using a sparse target and custom loss functions.
* Using NNs for decimation/interpolation (sub-sampling) in time, to reduce the frame rate and hence bit rate.
* Extending VQ VAE to two stage vector quantisation.  Multi-stage VQ is commonly used in non-NN speech coding.
* The simulations (NN training and vector quantisation) works on mean square error in the log(Am) domain, which is equivalence to (i) variance (ii) and proportional to Spectral Distortion (SD) in dB^2 - which is very closely correlated to subjective quality.  It trains in dB.

## Scripts

| Script | Description | Useful? |
| --- | --- | --- |
| codec2_model.py | Reading and writing Codec 2 model parameters | - |
| eband_train.py | Constant rate K to timing varying rate L, using LPCNet style K=14 vectors | No |
| ebanddec_train.py | Constant rate K to timing varying rate L, with decimation in time | No |
| eband_out.py | Generates Codec 2 output from NN trained in eband_train/ebanddec_train | No |
| newamp1_train.py | Similar to eband_train.py, constant rate K to timing varying rate L, using Codec 2 newamp1 K=20 vectors | No |
| vq_pager.py | Step through output rate K vectors | Yes |
| vq_vae_demo.py | Simple demo of VQ, nice visualisation of training in action | Cool demo | 
| vq_vae_demo_2stage.py | vq_vae_demo.py extended to two stage VQ | Cool demo | 
| vq_vae_ratek.py | Single stage VQ-VAE with single Dense layer | No |
| vq_vae_ratek_conv1d.py | Two stage VQ-VAE with two conv1D layers | Yes, reasonable spectral distortion, cool plots |

## Amplitude Sample Rate Conversion Using Neural Nets

Codec 2 models speech as a harmonic series of sine waves, each with it's own frequency, amplitude and phase.  The frequencies are approximated at harmonics of the pitch or fundamental frequency.  A reasonable model of the phases can be recovered from the amplitudes.

Accurate representation of the sine wave amplitudes {Am} m=1...L is important for good quality speech.  The number of amplitudes in each frame L is dependent on the pitch L=P/2, which is time varying between (typically between L=10 and L=80).  However for transmission at a fixed bit rate, a fixed number of parameters is desirable.

In earlier Codec 2 modes such as 3200 down to 1200, the amplitudes were represented using a fixed number of Linear Prediction Coefficients.  In more recent modes such as 700C, the amplitudes Am are resampled to a fixed sample rate (K=20), and vector quantised for transmission.  At the decoder, the rate K amplitude samples are resampled back to rate L for synthesis.  The K=20 vectors use mel spaced sampling so these vectors are similar to the mel-spaced MFCCs used by the NN community.

Some of the scripts in this repo explores the use of Neural Networks (NN) in resampling between rate K and rate L.  


