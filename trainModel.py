import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from conversion import spectrogram_to_image
from configparser import ConfigParser

def get_model(sliceSize, hiddenLayers, lossFunc, learningRate):
    global verbose

    # Create a simple model.
    inputs1 = keras.Input(shape=(257*sliceSize,))
    inputs2 = keras.layers.BatchNormalization(momentum=0.8)(inputs1)
    layers = [inputs2]

    for i, size in enumerate(hiddenLayers):
        layers.append(keras.layers.Dense(size, activation=tf.nn.leaky_relu, name=f"dense_{i+1}", kernel_initializer="random_uniform")(layers[i]))

    outputs = keras.layers.Dense(257*sliceSize,activation="sigmoid")(layers[-1])
    model = keras.Model(inputs1, outputs)
    opt = keras.optimizers.Adam(lr=learningRate)
    model.compile(optimizer=opt, loss=lossFunc)

    if verbose:
        print(model.get_weights())

    return model

def Main():
    global verbose

    config = ConfigParser()
    config.read('config.ini')

    modelName = config['MISC']['modelName']
    verbose = eval(config['MISC']['verbose'])

    sliceSize = int(config['Structure']['sliceSize'])
    hiddenLayers = [int(i) for i in config['Structure']['hiddenLayers'].split(',')]

    learningRate = float(config['Advanced']['learningRate'])
    lossFunc = config['Advanced']['lossFunc']
    batchSize = int(config['Advanced']['batchSize'])

    print('\n====================')
    print('Loaded from configuration file:\n')
    print('MISC:')
    print('\tModel Name:', modelName)
    print('Structure:')
    print('\tSlice Size:', sliceSize)
    print('\tHidden Layers:', hiddenLayers)
    print('Advanced:')
    print('\tLearning Rate:', learningRate)
    print('\tLoss Function:', lossFunc)
    print('Batch Size:', batchSize)
    print('====================\n')

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

    if os.path.isdir(modelName):
        model = keras.models.load_model(modelName)
    else:
        model = get_model(sliceSize, hiddenLayers, lossFunc, learningRate)

    try:
        model.fit(inputData, targetData,batchSize,10000)
    except KeyboardInterrupt:
        pass
    finally:
        print('\n\n/////////////////////////\n')
        print("DO NOT CLOSE -- MODEL SAVING!!!")
        print('\n/////////////////////////')
        model.save(modelName)
