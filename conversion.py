import librosa
import numpy as np
from PIL import Image
from scipy.io.wavfile import write as waveWrite
import os

def audio_to_spectrogram(filename, output):
    y, sr = librosa.load(filename)
    length = 2**13
    y = y[:length]

    window_size = 512
    window = np.hanning(window_size)
    hop_length = 64
    stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=hop_length, window=window)
    stft = stft.real
    stft = stft - stft.min()
    stft /= stft.max()
    out = 2 * np.abs(stft) / np.sum(window)

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

if __name__ == '__main__':
    for zone in ['input', 'output']:
        for file in os.listdir('training/'+zone):
            if file.split('.')[-1] == 'wav':
                audio_to_spectrogram('training/'+zone+'/'+file, '_training/'+zone+'/'+'.'.join(file.split('.')[:-1]))
    for file in os.listdir('use'):
        if file.split('.')[-1] == 'wav':
            audio_to_spectrogram('use/'+file, '_use/'+'.'.join(file.split('.')[:-1]))
