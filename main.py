import numpy as np
import tensorflow as tf
from tensorflow import keras
from conversion import spectrogram_to_audio, spectrogram_to_image
import os

reconstructed_model = keras.models.load_model("my_model")
for file in os.listdir('_use'):
    if file.split('.')[-1] == 'npy':
        test_input = np.array([np.load('_use/'+file).flatten()])
        out = reconstructed_model.predict(test_input)
        out.shape = 513, 129
        spectrogram_to_image(out, 'output/'+'.'.join(file.split('.')[:-1])+'Converted')
        spectrogram_to_audio(out, 'output/'+'.'.join(file.split('.')[:-1])+'Converted'+'.wav', 64, 22050)
