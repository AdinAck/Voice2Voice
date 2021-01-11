import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)

def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(33153,))
    x1 = keras.layers.Dense(1000, activation="sigmoid", name="dense_1")(inputs)
    # x2 = keras.layers.Dense(40000, activation="relu", name ="dense_2")(x1)
    outputs = keras.layers.Dense(33153,activation="sigmoid")(x1)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=0.000005)
    model.compile(optimizer=opt, loss="mean_squared_error")
    print(model.get_weights())
    return model


#model = get_model()
model = keras.models.load_model("my_model")

# Train the model.
inputData = []
for inputDataset in os.listdir('_training/input'):
    for segment in os.listdir(f'_training/input/{inputDataset}'):
        inputData.append(np.load(f'_training/input/{inputDataset}/{segment}').flatten())

inputData = np.asarray(inputData)

targetData = []
for targetDataset in os.listdir('_training/output'):
    for segment in os.listdir(f'_training/output/{targetDataset}'):
        targetData.append(np.load(f'_training/output/{targetDataset}/{segment}').flatten())

targetData = np.asarray(targetData)

print(inputData.shape, targetData.shape)
model.fit(inputData, targetData,4,100)

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
