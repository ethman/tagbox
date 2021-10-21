from sota_music_taggers.predict import MusicTagger
import torch
from torch import nn, optim
import torchaudio
from tqdm import trange

from .utils import JUKEBOX_SAMPLE_RATE, TAGGER_SR
from .utils import setup_jbx, encode, decode, make_masked_audio, to_np


disp = lambda a: float(to_np(a))


def run_tagbox(audio, labels, n_steps, step_size, device,
               model_types, training_data, mask_audio=False, n_fft=None, vqvae=None):
    """Run TagBox on audio file."""
    labels = labels.to(device)

    if vqvae is None:
        vqvae = setup_jbx('5b', device)
    vqvae.bottleneck.train()

    resampler = torchaudio.transforms.Resample(JUKEBOX_SAMPLE_RATE, TAGGER_SR).to(device)

    if not isinstance(model_types, list):
        model_types = [model_types]

    taggers = []
    for m in model_types:
        t = MusicTagger(m, training_data, return_feats=False)
        t.model.to(device)
        t.model.eval().requires_grad_(False)
        taggers.append(t)

    if mask_audio:
        if n_fft is None:
            n_fft = [2048, 1024, 512]
        elif not isinstance(n_fft, list):
            n_fft = [n_fft]

    audio = vqvae.decode(vqvae.encode(audio))  # pass thru once -> JBX will resize
    orig_audio = audio.clone()

    encoded_audio = encode(vqvae, audio)
    encoded_audio = [e.detach().requires_grad_(True) for e in encoded_audio]

    optimizer = optim.Adam(encoded_audio, lr=step_size)

    label_loss = nn.BCELoss()

    jbxd_audio = None
    hist_loss = []

    with trange(n_steps) as t:
        for _ in t:

            optimizer.zero_grad()

            zs, xs_quantised, _, _ = vqvae.bottleneck(encoded_audio)
            jbxd_audio = decode(vqvae, xs_quantised)
            jbxd_audio = jbxd_audio[-1]

            loss_dict = {}
            if mask_audio:
                for fft_size in n_fft:
                    masked_audio = make_masked_audio(orig_audio, jbxd_audio, fft_size)
                    for tagger in taggers:
                        pred_labels = tagger(resampler(masked_audio.squeeze()))
                        pred_labels = torch.mean(pred_labels, dim=0, keepdims=True)
                        loss_dict[f'{tagger}_{fft_size}'] = label_loss(pred_labels, labels)
            else:
                for tagger in taggers:
                    pred_labels = tagger(resampler(jbxd_audio.squeeze()))
                    pred_labels = torch.mean(pred_labels, dim=0, keepdims=True)
                    loss_dict[f'{tagger}'] = label_loss(pred_labels, labels)

            total_loss = sum(loss_dict.values()) / len(loss_dict)
            hist_loss.append({k: to_np(v) for k, v in loss_dict.items()})
            total_loss.backward()
            optimizer.step()

            t.set_postfix(loss=disp(total_loss))

    results = {
        'jbxd_audio': to_np(jbxd_audio),
        'orig_audio': to_np(orig_audio),
        'hist_loss': hist_loss,
    }

    if mask_audio:
        jbxd_masked = make_masked_audio(orig_audio, jbxd_audio, fft_size)
        results.update({
            'jbxd_masked': jbxd_masked,
            'jbxd_diff': to_np(orig_audio) - to_np(jbxd_masked)
        })

    return results
