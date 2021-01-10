import numpy as np
from tensorflow import keras
model = keras.models.load_model("my_model")
print(model.get_weights()[0][100])
print(model.get_weights()[1][100])
for i in model.get_weights()[1]:
    print(i)
#print(model.get_weights())
#
