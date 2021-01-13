import numpy as np
import tensorflow as tf
from tensorflow import keras
from conversion import spectrogram_to_image
import os

# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)

def get_model(sliceSize):
    # Create a simple model.
    inputs = keras.Input(shape=(257*sliceSize,))
    x1 = keras.layers.Dense(1000, activation="relu", name="dense_1")(inputs)
    x2 = keras.layers.Dense(1000, activation="relu", name ="dense_2")(x1)
    outputs = keras.layers.Dense(257*sliceSize,activation="sigmoid")(x2)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss="mean_squared_error")
    # print(model.get_weights())
    return model


sliceSize = 512

# Train the model.
inputData = []
targetData = []
for inputDataset, targetDataset in zip(os.listdir('_training/input'), os.listdir('_training/output')):
    iterator = zip(
        [file for file in
            os.listdir(f'_training/input/{inputDataset}')
                if file.split('.')[-1] == 'npy'],
        [file for file in
            os.listdir(f'_training/output/{targetDataset}')
                if file.split('.')[-1] == 'npy']
    )

    for inSegment, outSegment in iterator:
        loadIn = np.load(f'_training/input/{inputDataset}/{inSegment}')
        loadIn *= loadIn/loadIn.max()
        if loadIn.shape[1] < sliceSize:
            print(f'Dataset "{inputDataset}" will be omitted from training because it is too short.')
            continue

        loadTarget = np.load(f'_training/output/{targetDataset}/{outSegment}')
        loadTarget *= loadTarget/loadTarget.max()
        if loadTarget.shape[1] < sliceSize:
            print(f'Dataset "{targetDataset}" will be omitted from training because it is too short.')
            continue

        for i, j in zip(range(np.size(loadIn, 1)//sliceSize),range(np.size(loadTarget, 1)//sliceSize)):
            a = loadIn[:,i*sliceSize:(i+1)*sliceSize]
            b = loadTarget[:,j*sliceSize:(j+1)*sliceSize]

            inputData.append(a.flatten())
            targetData.append(b.flatten())

inputData = np.asarray(inputData)

targetData = np.asarray(targetData)

print(inputData.shape, targetData.shape)

model = get_model(sliceSize)
# model = keras.models.load_model("my_model")

for _ in range(100):
    model.fit(inputData, targetData,len(inputData),1000, use_multiprocessing=True)
    print("DO NOT CLOSE -- MODEL SAVING!!!")
    model.save("my_model")


# Calling `save('my_model')` creates a SavedModel folder `my_model`.


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
