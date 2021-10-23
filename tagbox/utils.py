
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F


from jukebox.make_models import make_vqvae, MODELS
from jukebox.hparams import setup_hparams, Hyperparams, DEFAULTS


TAGGER_SR = 16000  # Hz
JUKEBOX_SAMPLE_RATE = 44100  # Hz


def setup_jbx(model, device, levels=3, sample_length=1048576):
    """Sets up the Jukebox VQ-VAE."""
    vqvae = MODELS[model][0]
    hparams = setup_hparams(vqvae, dict(sample_length=sample_length,
                                        levels=levels))

    for default in ["vqvae", "vqvae_conv_block", "train_test_eval"]:
        for k, v in DEFAULTS[default].items():
            hparams.setdefault(k, v)

    hps = Hyperparams(**hparams)
    return make_vqvae(hps, device)


def audio_for_jbx(audio, trunc_sec=None, device='cuda'):
    """Readies an audio array for Jukebox."""
    if audio.ndim == 1:
        audio = audio[None]
        audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    audio = audio.flatten()
    if trunc_sec is not None:
        audio = audio[: int(JUKEBOX_SAMPLE_RATE * trunc_sec)]

    return torch.tensor(audio, device=device)[None, :, None]


def load_audio_for_jbx(path, offset=0.0, dur=None, trunc_sec=None, device='cuda'):
    """Loads a path for use with Jukebox."""
    audio, sr = librosa.load(path, sr=None, offset=offset, duration=dur)

    if sr != JUKEBOX_SAMPLE_RATE:
        audio = librosa.resample(audio, sr, JUKEBOX_SAMPLE_RATE)

    return audio_for_jbx(audio, trunc_sec, device=device)


def make_labels(l, n_classes=50):
    """Turns a list of ints into a multi-hot label vector."""
    if isinstance(l, int):
        l = [l]
    return torch.sum(F.one_hot(torch.tensor(l), num_classes=n_classes).type(torch.float32),
                     dim=0, keepdims=True)


def encode(vqvae, x):
    """Encode audio, `x`, to an unquantized embedding using `vqvae`."""
    x_in = vqvae.preprocess(x)
    xs = []
    for level in range(vqvae.levels):
        encoder = vqvae.encoders[level]
        x_out = encoder(x_in)
        xs.append(x_out[-1])

    return xs


def decode(vqvae, xs_quantized, level=0):
    """Decode quantized codes, `xs_quantized` back to audio."""
    # TODO: Don't pass thru all 3 layers, it's hacky! :D
    x_outs = []
    for level in range(vqvae.levels):
        decoder = vqvae.decoders[level]
        x_out = decoder(xs_quantized[level:level + 1], all_levels=False)
        x_outs.append(x_out)

    if level is None:
        return x_outs
    return x_outs[level]


def make_masked_audio(input_audio, jbx_audio, n_fft):
    """Use Jukebox's audio to mask the input_audio."""
    eps = 1e-8

    window = torch.from_numpy(np.sqrt(scipy.signal.get_window('hann', n_fft)))
    window = window.to(input_audio.device).to(input_audio.dtype)

    stft = lambda x: torch.stft(x, n_fft, hop_length=n_fft // 2,
                                window=window,
                                return_complex=True)

    input_stft = stft(input_audio.squeeze())
    input_spec, input_phase = torch.abs(input_stft), torch.angle(input_stft)
    jbx_spec = torch.abs(stft(jbx_audio.squeeze()))
    mask = jbx_spec / (torch.maximum(input_spec, jbx_spec) + eps)

    masked_spec = mask * input_spec
    masked_stft = masked_spec * torch.exp(1j * input_phase)
    masked_audio = torch.istft(masked_stft, n_fft, hop_length=n_fft // 2,
                               window=window,
                               length=jbx_audio.shape[-1]).unsqueeze(0)
    return masked_audio


def to_np(a):
    """Convert tensor to numpy."""
    if not isinstance(a, np.ndarray):
        return a.clone().squeeze().detach().cpu().numpy()
    return a


def display_spec(mix, gt, est, est_other, gt_other, save_path,
                 sr=JUKEBOX_SAMPLE_RATE, y_ax='mel'):
    """Display spectrograms."""
    npify = lambda sig: sig.audio_data.squeeze()
    plt.close('all')
    all_sigs = {
        'Mix': mix,
        'Ground Truth': gt,
        'Estimated': est,
        'Est. Other': est_other,
        'GT Other': gt_other,
        'Mix - Est.': mix - est
    }

    fig, axes = plt.subplots(nrows=3, ncols=2, sharey=True, figsize=(12, 18))
    for i, (title, sig) in enumerate(all_sigs.items()):
        audio = npify(sig)
        ax = axes.flatten()[i]
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(spec, y_axis=y_ax, x_axis='time', sr=sr, ax=ax)

        ax.set(title=title)
        ax.label_outer()

    plt.savefig(save_path)
