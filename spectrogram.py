import os
import sys
import torch
import speechbrain as sb
from speechbrain.processing.features import spectral_magnitude
import matplotlib
import matplotlib.pyplot as plt
import re

__all__ = ["save_spec"]

# matplotlib.use('pgf')
# plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams['axes.formatter.use_mathtext'] = True

# plt.rcParams.update({
#     "text.usetex": True,
# })
# plt.rc('axes', unicode_minus=False)

# plt.rcParams["pdf.use14corefonts"] = True

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
# plt.rcParams['axes.labelsize'] = 40
# plt.tick_params(labelsize=25)
plt.rcParams['font.size'] = 30

aspect = 0.65


def generate_spec(filename: str):
    if not hasattr(generate_spec, "stft"):
        generate_spec.stft = sb.processing.features.STFT(sample_rate=16000, win_length=32, hop_length=16, n_fft=512, window_fn=torch.hamming_window)
    sig = sb.dataio.dataio.read_audio(filename).unsqueeze(0)
    feats = generate_spec.stft(sig).squeeze(0)
    # feats = spectral_magnitude(feats, power=0.5)

    # feats = torch.log1p(feats)
    return feats

def save_spec(in_folder, out_folder):
    for filename in os.listdir(in_folder):
        if not re.match(r".+\.wav", filename) is None:
            basename = filename.split(".")[0]
            feats = generate_spec(os.path.join(in_folder, filename))
            mag = spectral_magnitude(feats, power=0.5)
            mag = torch.log1p(mag).T
            mag = torch.log10(mag)
            # phase = torch.atan2(feats[:, :, 1], feats[:, :, 0]).T

            plt.imshow(mag, origin="lower", aspect=aspect)
            # plt.title(f"Spectrogram of \\verb|{filename}|")
            plt.xlabel("Time frames")
            plt.ylabel("Frequency bins")
            plt.colorbar(label="dB")
            plt.savefig(f"{out_folder}/{basename}-mag.png", bbox_inches="tight")
            # plt.show(block=False)
            plt.close()


if __name__ == "__main__":
    directory = "." if len(sys.argv) <= 1 else sys.argv[1]
    output_dir = "." if len(sys.argv) <= 2 else sys.argv[2]
    save_spec(directory, output_dir)
