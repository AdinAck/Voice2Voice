import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from conversion import spectrogram_to_audio, spectrogram_to_image
from tqdm import trange
from configparser import ConfigParser

def Main():
    config = ConfigParser()
    config.read('config.ini')

    modelName = config['MISC']['modelName']
    verbose = eval(config['MISC']['verbose'])

    reconstructed_model = keras.models.load_model(modelName)
    if verbose:
        print(reconstructed_model.get_weights())

    sliceSize = int(config['Structure']['sliceSize'])

    for dir in os.listdir('_use'):
        tmp = []
        for file in os.listdir(f'_use/{dir}'):
            if file.split('.')[-1] == 'npy':
                test_input = np.load(f'_use/{dir}/{file}')
                if verbose:
                    print('Input shape:')
                    print(test_input.shape)
                for i in trange(test_input.shape[1]//sliceSize, unit='spec'):
                    out = reconstructed_model.predict(np.array([test_input[:,i*sliceSize:(i+1)*sliceSize].flatten()]))
                    out -= .5
                    out *= 10
                    out.shape = 257, sliceSize
                    tmp.append(out)

        final = tmp[0]
        for arr in tmp[1:]:
            final = np.concatenate((final, arr), axis=1)

        if verbose:
            spectrogram_to_image(final, f'output/{dir}Converted')
            np.save(f'output/{dir}Converted.npy', final)
        spectrogram_to_audio(final, f'output/{dir}Converted'+'.wav', 64, 22050)
