import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from conversion import spectrogram_to_audio, spectrogram_to_image
from tqdm import tqdm

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)

reconstructed_model = keras.models.load_model("my_model")

for dir in os.listdir('_use'):
    tmp = []
    for file in tqdm(os.listdir(f'_use/{dir}'), unit='spec'):
        if file.split('.')[-1] == 'npy':
            test_input = np.array([np.load(f'_use/{dir}/{file}').flatten()])
            #test_input = np.array([np.ones((257*129))+50])
            #print(test_input.shape)
            out = reconstructed_model.predict(test_input)
            print(np.average(out))
            out -= out.min()
            out *= 160/out.max()
            out.shape = 257, 129
            tmp.append(out)

    final = tmp[0]
    for arr in tmp[1:]:
        final = np.concatenate((final, arr), axis=1)

    spectrogram_to_image(final, f'output/{dir}Converted')
    spectrogram_to_audio(final, f'output/{dir}Converted'+'.wav', 64, 22050)
