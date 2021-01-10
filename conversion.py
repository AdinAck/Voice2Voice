import librosa
import numpy as np
from PIL import Image
from scipy.io.wavfile import write as waveWrite
import os

def audio_to_spectrogram(filename, output):
    y, sr = librosa.load(filename)
    length = 2**13
    y = y[:length]

    window_size = 1024
    window = np.hanning(window_size)
    hop_length = 64
    stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=hop_length, window=window)
    out = 2 * np.abs(stft) / np.sum(window)
    stft = stft - stft.min()

    if output not in os.listdir():
        os.mkdir(output)
    np.save(output+'.npy', stft)

    spectrogram_to_image(out, output)

    return hop_length, sr

def spectrogram_to_audio(arr, output, hop_length, sr):
    outDir = '.'.join(output.split('.')[:-1])
    audio = librosa.core.istft(arr, hop_length=hop_length)
    waveWrite(output, sr, audio)

def spectrogram_to_image(spec, name):
    img = spec.copy()
    img *= 255/img.max()
    img -= img.min()
    img = np.flip(img, 0)
    Image.fromarray(img).convert('RGB').save(name+'.png')
