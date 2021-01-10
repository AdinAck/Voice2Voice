import numpy as np
import tensorflow as tf
from tensorflow import keras
from conversion import spectrogram_to_audio, spectrogram_to_image

reconstructed_model = keras.models.load_model("my_model")
test_input = np.array([np.load('adinSpec2.npy').flatten()])
out = reconstructed_model.predict(test_input)
out.shape = 513, 129
spectrogram_to_image(out, 'artini')
spectrogram_to_audio(out, 'testOut.wav', 64, 22050)
