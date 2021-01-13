import numpy as np
import tensorflow as tf
from tensorflow import keras
import wandb
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
    inputs1 = keras.Input(shape=(257*sliceSize,))
    inputs2 = keras.layers.BatchNormalization(momentum=0.8)(inputs1)
    x1 = keras.layers.Dense(1500, activation=tf.nn.leaky_relu, name="dense_1", kernel_initializer="random_uniform")(inputs2)
    x2 = keras.layers.Dense(2500, activation=tf.nn.leaky_relu, name="dense_2", kernel_initializer="random_uniform")(x1)
    x3 = keras.layers.Dense(1500, activation=tf.nn.leaky_relu, name="dense_3", kernel_initializer="random_uniform")(x2)
    outputs = keras.layers.Dense(257*sliceSize,activation=None)(x3)
    model = keras.Model(inputs1, outputs)
    opt = keras.optimizers.Adam(lr=0.000147)
    model.compile(optimizer=opt, loss="mean_squared_error")
    # print(model.get_weights())
    return model

wandb.init(project='Voice2Voice', name='Voice2Voice')

sliceSize = 256

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

try:
    for epoch in range(100000):
        # print(model.get_weights())
        model.fit(inputData, targetData,4,1000, use_multiprocessing=True)
        # loss = model.train_on_batch(inputData, targetData)
        # wandb.log({"Loss": loss})
except KeyboardInterrupt:
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
