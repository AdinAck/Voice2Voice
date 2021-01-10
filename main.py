import numpy as np
import tensorflow as tf
from tensorflow import keras
from conversion import spectrogram_to_audio, spectrogram_to_image
import os

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

reconstructed_model = keras.models.load_model("my_model")
for file in os.listdir('_use'):
    if file.split('.')[-1] == 'npy':
        test_input = np.array([np.load('_use/'+file).flatten()])
        #test_input = np.array([np.ones((257*129))+50])
        #print(test_input.shape)
        out = reconstructed_model.predict(test_input)*160
        out.shape = 257, 129

        spectrogram_to_image(out, 'output/'+'.'.join(file.split('.')[:-1])+'Converted')
        spectrogram_to_audio(out, 'output/'+'.'.join(file.split('.')[:-1])+'Converted'+'.wav', 64, 22050)
