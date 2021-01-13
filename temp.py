import numpy as np
from conversion import spectrogram_to_audio

spectrogram_to_audio(np.load('_training/input/adinHi/0.npy')*160, 'test.wav', 64, 22050)
