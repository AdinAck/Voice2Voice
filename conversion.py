import os, shutil
import numpy as np
from PIL import Image
import librosa
from scipy.io.wavfile import write as waveWrite
from tqdm import tqdm

def sigmoid(x):
    return 1/(1+np.exp(-x))

def audio_to_spectrogram(filename, output):
    y, sr = librosa.load(filename)
    window_size = 512
    window = np.hanning(window_size)
    hop_length = 64

    if not os.path.isdir(output):
        os.mkdir(output)
    stft = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=hop_length, window=window)
    stft = stft.real
    stft = np.vectorize(sigmoid)(stft)

    np.save(f'{output}/{0}.npy', stft)

    spectrogram_to_image(stft, f'{output}/{0}')

    return hop_length, sr

def spectrogram_to_audio(arr, output, hop_length, sr):
    outDir = '.'.join(output.split('.')[:-1])
    audio = librosa.core.istft(arr, hop_length=hop_length)
    waveWrite(output, sr, audio)

def spectrogram_to_image(spec, name):
    img = spec.copy()
    img -= img.min()
    img *= 255/img.max()
    img = np.flip(img, 0)
    Image.fromarray(img).convert('RGB').save(name+'.png')

def Main(flush=False):
    if flush:
        print("Flushing directories...")
        dirs = [dir for dir in os.listdir() if dir in ['_training', '_use']]
        for dir in dirs:
            shutil.rmtree(dir)
        print('Done.')
    else:
        print('Populating directories...')
        dirs = os.listdir()
        stop = False
        for dir in ['_training', '_use', 'training', 'use', 'output']:
            if dir not in dirs:
                if dir in ['training', 'use']:
                    stop = True
                os.mkdir(dir)

        for dir in ['_training', 'training']:
            dirs = os.listdir(dir)
            if 'input' not in dirs:
                os.mkdir(f'{dir}/input')
            if 'output' not in dirs:
                os.mkdir(f'{dir}/output')

        if stop:
            print('Created missing folders, please load data.')
            print('Exiting...')
            exit()

        trainFiles = ['input/'+file for file in os.listdir('training/input') if '.'.join(file.split('.')[:-1]) not in os.listdir('_training/input')]+\
                     ['output/'+file for file in os.listdir('training/output') if '.'.join(file.split('.')[:-1]) not in os.listdir('_training/output')]
        useFiles = [file for file in os.listdir('use') if '.'.join(file.split('.')[:-1]) not in os.listdir('_use')]

        print('Converting...\n')
        progress = tqdm(total=len(trainFiles)+len(useFiles), unit=' sgmnts')

        for file in trainFiles:
            if file.split('.')[-1] == 'wav':
                audio_to_spectrogram(f'training/{file}', f'_training/'+'.'.join(file.split('.')[:-1]))
                progress.update(1)
        for file in useFiles:
            if file.split('.')[-1] == 'wav':
                audio_to_spectrogram('use/'+file, '_use/'+'.'.join(file.split('.')[:-1]))
                progress.update(1)
