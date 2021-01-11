import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(257,))
    x1 = keras.layers.Dense(1000, activation="sigmoid", name="dense_1")(inputs)
    # x2 = keras.layers.Dense(40000, activation="relu", name ="dense_2")(x1)
    outputs = keras.layers.Dense(257,activation="sigmoid")(x1)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(lr=0.0005)
    model.compile(optimizer=opt, loss="mean_squared_error")
    # print(model.get_weights())
    return model


model = get_model()
#model = keras.models.load_model("my_model")

# Train the model.
inputData = []
targetData = []
for inputDataset, targetDataset in zip(os.listdir('_training/input'), os.listdir('_training/output')):
    for inSegment, outSegment in zip(os.listdir(f'_training/input/{inputDataset}'), os.listdir(f'_training/output/{targetDataset}')):
        loadIn = np.load(f'_training/input/{inputDataset}/{inSegment}').flatten()
        loadIn *= loadIn/loadIn.max()

        loadTarget = np.load(f'_training/output/{targetDataset}/{outSegment}').flatten()
        loadTarget *= loadTarget/loadTarget.max()

        for collumnIn, collumnTarget in zip(loadIn,loadTarget):
            inputData.append(collumnIn)
            targetData.append(collumnTarget)
size = len(inputData)//257
inputData = np.asarray(inputData).reshape((size,257))

targetData = np.asarray(targetData).reshape((size,257))

print(inputData.shape, targetData.shape)
model.fit(inputData, targetData,len(inputData),2000, use_multiprocessing=True)

print("DO NOT CLOSE -- MODEL SAVING!!!")

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")


#reconstructed_model.predict(test_input)



# It can be used to reconstruct the model identically.
#reconstructed_model = keras.models.load_model("my_model")

# Let's check:
# np.testing.assert_allclose(
#     model.predict(test_input), reconstructed_model.predict(test_input)
# )

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
# reconstructed_model.fit(test_input, test_target)
