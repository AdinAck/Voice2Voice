# Description
A Python neural network made with TensorFlow that converts one person's voice into another. The network is trained on audio files of person A's voice that person B needs to replicate to the best degree.
***
# Setup

## Prerequisites
 - Python 3.7 or greater (64-bit)
 - Python package requirements
 - NVIDIA Graphics card
 - Required Drivers and Development Tools

## 1. Package Requirements
To install the required python libraries, simply execute the following command in the repository working directory.

`pip install -r requirements.txt`

## 2. (**Highly Recommended**) Graphics Card Utilization

### A) Required Drivers and Development Tools
 - Latest NVIDIA GPU Drivers
 - CUDA Toolkit (v11.0 Update 1)
 - cuDNN SDK 8.0.4 for CUDA Toolkit v11.0

### B) cuDNN
Create a folder named *tools* under C:\\. Drag the *cuda* folder from the cuDNN zip into the tools folder.

### C) Appending %PATH%
Add the following paths to the PATH system environment variable:
 - `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\bin`
 - `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\extras\\CUPTI\\lib64`
 - `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\include`
 - `C:\\tools\\cuda\\bin`

## 3. (Optional) Configuration
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

## 4. Data

### Create folders
`Voice2Voice.py -l`

Running `load_data` for the first time creates the `training` and `use` folders.

### Populate folders
#### Supplying data
 - Only **wav** files are supported.
 - Training files must be in order, corresponding with the files in the other folder.

#### Training/Input
Place person A's audio recordings into the folder.

#### Training/Output
Place person B's audio recordings in an order corresponding to person A's audio recordings. (Recommended naming example: "helloJohn.wav")

#### Use Folder
Place files to be used by model to perform voice conversion. (Suggest using training files as preliminary testing of model)

### Convert
`Voice2Voice.py -l`

Running `load_data` for the second time converts the contents of `inputs`, `outputs`, and `use` into processable files.

## 5. Train Model
`Voice2Voice.py -t`

If let Run for 10,000 epochs or **<Ctrl + C>** is pressed, the model will be saved **DO NOT CLOSE TERMINAL UNTIL MODEL SAVED**.

## 6. Prediction Time!!!
`Voice2Voice.py -p`

Use the model assigned in `config.ini` to convert voices from `use` folder and place them in `output`.


## Usage
`Voice2Voice.py [-l | --load_data] [-f | --flush_data] [-t | --train] [-p | --predict]`

| Argument | Description |
| -------- | ----------- |
| [-l \| load_data] | Load audio files in `training` and `use` folders to be usable by the model. This will also create the `training` and `use` folders if they are not present.|
| [-f \| flush_data] | Delete all converted data. |
| [-t \| --train] | Create a new model to be trained, or continue training an existing model (dependent on the `modelName` attribute in `config.ini`). Exit training and save model by interrupting the process **<Ctrl + C>**. |
| [-p \| --predict] | Load model specified by `modelName` in `config.ini` and predict audio output given audio files in `use` folder. |
