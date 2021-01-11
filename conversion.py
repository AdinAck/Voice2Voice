print('Loading libraries...')

import os
import numpy as np
from PIL import Image
import librosa
from scipy.io.wavfile import write as waveWrite
from tqdm import tqdm

def audio_to_spectrogram(filename, output):
    y, sr = librosa.load(filename)
    length = 2**13
    window_size = 512
    window = np.hanning(window_size)
    hop_length = 64

    if not os.path.isdir(output):
        os.mkdir(output)

    for i in range(max(1,len(y)//length-1)):
        stft  = librosa.core.spectrum.stft(y[i*length:(i+1)*length], n_fft=window_size, hop_length=hop_length, window=window)
        stft = stft.real
        stft = stft - stft.min()
        stft /= stft.max()
        out = 2 * np.abs(stft) / np.sum(window)

        np.save(f'{output}/{i}.npy', stft)

        # spectrogram_to_image(out, f'{output}/{i}')

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
    print('Populating directories...')
    dirs = os.listdir()
    for dir in ['_training', '_use', 'training', 'use', 'output']:
        if dir not in dirs:
            os.mkdir(dir)

    for dir in ['_training', 'training']:
        dirs = os.listdir(dir)
        if 'input' not in dirs:
            os.mkdir(f'{dir}/input')
        if 'output' not in dirs:
            os.mkdir(f'{dir}/output')

    trainFiles = ['input/'+file for file in os.listdir('training/input')]+\
                 ['output/'+file for file in os.listdir('training/output')]
    useFiles = os.listdir('use')

    print('Converting...\n')
    progress = tqdm(total=len(trainFiles)+len(useFiles), unit='spec')

    for file in trainFiles:
        if file.split('.')[-1] == 'wav':
            audio_to_spectrogram(f'training/{file}', f'_training/'+'.'.join(file.split('.')[:-1]))
            progress.update(1)
    for file in useFiles:
        if file.split('.')[-1] == 'wav':
            audio_to_spectrogram('use/'+file, '_use/'+'.'.join(file.split('.')[:-1]))
            progress.update(1)
