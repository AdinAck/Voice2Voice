# Prerequisites
 - Python 3.7 or greater (64-bit)
 - Python package requirements
 - NVIDIA Graphics card
 - Required Drivers and Development Tools

## Package Requirements
To install the required python libraries, simply execute the following command in the repository working directory.

`pip install -r requirements.txt`

## Required Drivers and Development Tools
 - Latest NVIDIA GPU Drivers
 - CUDA Toolkit (v11.0 Update 1)
 - cuDNN SDK 8.0.4 for CUDA Toolkit v11.0

### cuDNN
Create a folder named *tools* under C:\\. Drag the *cuda* folder from the cuDNN zip into the tools folder.

### Appending %PATH%
Add the following paths to the PATH system environment variable:
 - `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\bin`
 - `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\extras\\CUPTI\\lib64`
 - `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\include`
 - `C:\\tools\\cuda\\bin`

# Configuration
In the `config.ini` file, there are attributes to be changed for configuring the model.

| Section   | Parameter | Description |
| --------- | --------- | ----------- |
| MISC | modelName | The name of the model to be created or loaded. |
| | verbose | If set to True, additional information will be printed while running, along with additional files for easy debugging. |
| Structure | sliceSize | The number of time samples to be used in the input. (if sliceSize exceeds the length of an audio clip, the audio clip will be omitted from the training data) |
| | hiddenLayers | A list of integers defining the size of each hidden layer. (should be formatted like so: a,b,c,d or a)
| Advanced | learningRate | The learning rate for the Adam optimizer. [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam#args) |
| | lossFunc | The loss function. [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile) |
| | batchSize | The batch size. [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit) |

## MISC
### modelName
The name of the model to be created or loaded.

### verbose
Set verbosity.

## Structure
### sliceSize
The number of time samples to be used in the input.

### hiddenLayers
A list of integers defining the size of each hidden layer.
(should be formatted like so: a,b,c,d or a)

## Advanced
### learningRate
The learning rate for the Adam optimizer. [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam#args)

### lossFunc
The loss function. [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile)

### batchSize
The batch size. [Tensorflow Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit)

# Usage
`Voice2Voice.py [-l | --load_data] [-f | --flush_data] [-t | --train] [-p | --predict]`

| Argument | Description |
| -------- | ----------- |
| [-l \| load_data] | Load audio files in `training` and `use` folders to be usable by the model. This will also create the `training` and `use` folders if they are not present.|
| [-f \| flush_data] | Delete all converted data. |
| [-t \| --train] | Create a new model to be trained, or continue training an existing model (dependent on the `modelName` attribute in `config.ini`). Exit training and save model by interrupting the process **<Ctrl + C>**. |
| [-p \| --predict] | Load model specified by `modelName` in `config.ini` and predict audio output given audio files in `use` folder. |

### load_data
Load audio files in `training` and `use` folders to be usable by the model.

This will also create the `training` and `use` folders if they are not present.

### flush_data
Delete all converted data.

### train
Create a new model to be trained, or continue training an existing model (dependent on the `modelName` attribute in `config.ini`).

Exit training and save model by interrupting the process (Ctrl + C).

### predict
Load model specified by `modelName` in `config.ini` and predict audio output given audio files in `use` folder.
