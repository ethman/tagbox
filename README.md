# TagBox


Steer OpenAI's Jukebox with Music Taggers!

**The closest thing we have to VQGAN+CLIP for music!**

## Unsupervised Source Separation By Steering Pretrained Music Models

Read the paper [here](https://arxiv.org/abs/2110.13071). Submitted to ICASSP 2022.

### Abstract

We showcase an unsupervised method that repurposes deep models trained for music
generation and music tagging for audio source separation, without any retraining.
An audio generation model is conditioned on an input mixture, producing a latent
encoding of the audio used to generate audio. This generated audio is fed to a
pretrained music tagger that creates source labels. The cross-entropy loss
between the tag distribution for the generated audio and a predefined distribution
for an isolated source is used to guide gradient ascent in the (unchanging)
latent space of the generative model. This system does not update the weights of
the generative model or the tagger, and only relies on moving through the
generative model's latent space to produce separated sources. We use OpenAI's
Jukebox as the pretrained generative model, and we couple it with four kinds of
pretrained music taggers (two architectures and two tagging datasets).
Experimental results on two source separation datasets, show this approach can
produce separation estimates for a wider variety of sources than any tested
supervised or unsupervised system. This work points to the vast and heretofore
untapped potential of large pretrained music models for audio-to-audio tasks
like source separation. 


## Try it yourself!


Run it yourself Colab notebook here: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ethman/tagbox)


## Example Output — Separation

*MUSDB18 and Slakh2100 examples coming soon!*  

Audio examples are not displayed on `https://github.com/ethman/tagbox`, please
click [here](https://ethman.github.io/tagbox/) to see the demo page.


TagBox excels in separating prominent melodies from within sparse mixtures.

### Wonderwall by Oasis - Vocal Separation

**Mixture**

<audio controls> <source src="examples/wonderwall/ww_mix.wav" type="audio/wav"> </audio>

**TagBox Output**

<audio controls> <source src="examples/wonderwall/ww_vox.wav" type="audio/wav"> </audio>

| hyperparam      | setting                  |
|-----------------|--------------------------|
| fft size(s)     |  512, 1024, 2048         |
| lr              |  10.0                    |
| steps           |  200                     |
| tagger model(s) |  fcn, hcnn, musicnn      |
| tagger data     |  MTAT                    |
| selected tags   |  All vocal tags          |

### Howl's Moving Castle, Piano & Violin Duet - Violin Separation

**Mixture**

<audio controls> <source src="examples/howls_castle/howl_mix.wav" type="audio/wav"> </audio>

**TagBox Output**

<audio controls> <source src="examples/howls_castle/howl_str.wav" type="audio/wav"> </audio>

| hyperparam      | setting                  |
|-----------------|--------------------------|
| fft size(s)     |  512, 1024, 2048         |
| lr              |  10.0                    |
| steps           |  100                     |
| tagger model(s) |  fcn, hcnn, musicnn      |
| tagger data     |  MTG-Jamendo             |
| selected tags   |  Violin                  |

### Smoke On The Water, by Deep Purple - Vocal Separation

**Mixture**

<audio controls> <source src="examples/smoke_on_the_water/soow_mix.wav" type="audio/wav"> </audio>

**TagBox Output**

<audio controls> <source src="examples/smoke_on_the_water/soow_vox.wav" type="audio/wav"> </audio>

| hyperparam      | setting                  |
|-----------------|--------------------------|
| fft size(s)     |  512, 1024, 2048         |
| lr              |  5.0                     |
| steps           |  200                     |
| tagger model(s) |  fcn, hcnn               |
| tagger data     |  MTAT                    |
| selected tags   |  All vocal tags          |


## Example Output - Improving Perceptual Output & "Style Transfer"

### Adding multiple FFT sizes helps with perceptual quality

Similar to multi-scale spectral losses, when we use masks with multiple FFT sizes
we notice that the quality of the output increases.

**Mixture**

<audio controls> <source src="examples/james_may_broken/jm_mix.wav" type="audio/wav"> </audio>

**TagBox with `fft_size=[1024]`**

Notice the warbling effects in the following example:

<audio controls> <source src="examples/james_may_broken/jm_ex1.wav" type="audio/wav"> </audio>


**TagBox with `fft_size=[1024, 2048]`**

Those warbling effects are mitigated by using two fft sizes:

<audio controls> <source src="examples/james_may_broken/jm_ex2.wav" type="audio/wav"> </audio>

These results, however, are not reflected in the SDR evaluation metrics.


### "Style Transfer"

Remove the masking step enables Jukebox to generate *any* audio that will optimize the
tag. In some situations, TagBox will pick out the melody and resynthesize it. But
it adds lots of artifacts, making it sound like the audio was recorded in a snowstorm.

**Mixture**

<audio controls> <source src="examples/james_may_dont_look_back/jm_mix.wav" type="audio/wav"> </audio>


**"Style Transfer"**

Here, we optimize the "guitar" tag without the mask:
<audio controls> <source src="examples/james_may_dont_look_back/jm_ex3_st.wav" type="audio/wav"> </audio>

## Cite

If you use this your academic research, please cite the following:

    @misc{manilow2021unsupervised,
      title={Unsupervised Source Separation By Steering Pretrained Music Models}, 
      author={Ethan Manilow and Patrick O'Reilly and Prem Seetharaman and Bryan Pardo},
      year={2021},
      eprint={2110.13071},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
    }
