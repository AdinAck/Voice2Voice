# import tensorflow as tf
#
# x = [1,1]
# y = [1]
#
# x0 = tf.constant(x,dtype=tf.float32)
# y0 = tf.constant(y,dtype=tf.float32)
#
# m1 = tf.Variable(tf.random_uniform([2,3],minval = 0.1,maxval=0.9,dtype=tf.float32))
# b1 = tf.Variable(tf.random_uniform([3,1],minval = 0.1,maxval=0.9,dtype=tf.float32))
# h1 = tf.sigmoid(tf.matmul(x0,m1) +b1)
#
# m2 = tf.Vaiable(tf.random_uniform([3,1],minval = 0.1,maxval=0.9,dtype=tf.float32))
# b2 = tf.Vaiable(tf.random_uniform([1],minval = 0.1,maxval=0.9,dtype=tf.float32))
# y_out = tf.sigmoid( tf.matmul( h1,m2 ) + b2 )
#
# loss = tf.reduce_sum( tf.square( y0 - y_out ) )
#
# train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(1):
#         sess.run(train)
#
#     results = sess.run([m1,b1,m2,b2,y_out,loss])
#     labels  = "m1,b1,m2,b2,y_out,loss".split(",")
#     for label,result in zip(*(labels,results)) :
#         print()
#         print(label)
#         print(result)
#
# print()
# Import `tensorflow`
# import os
# from PIL import Image
# def load_data(data_directory):
#     _, directis,_=os.walk(data_directory)
#     directories = [d for d in os.listdir(data_directory)
#                    if os.path.isdir(os.path.join(data_directory, d))]
#     ouputs = []
#     images = []
#     for d in directories:
#         ouput_directory = os.path.join(data_directory, d)
#         file_names = [os.path.join(ouput_directory, f)
#                       for f in os.listdir(ouput_directory)
#                       if f.endswith(".ppm")]
#         for f in file_names:
#             images.append(skimage.data.imread(f))
#             labels.append(int(d))
#     return images, labels
#
# def load(input_directory,ouput_directory):
#     _, directis,_=os.walk(data_directory)
#     _, directis,_=os.walk(data_directory)
#
#
#
# ROOT_PATH = "/your/root/path"
# train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
# test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")
#
# images, labels = load_data(train_data_directory)

# import tensorflow as tf
#
# # Initialize placeholders
# x = tf.placeholder(dtype = tf.float32, shape = [None, 129, 513])
# # Flatten the input data
# images_flat = tf.contrib.layers.flatten(x)
#
# # Fully connected layer
# logits1 = tf.contrib.layers.fully_connected(images_flat, 100000, tf.nn.relu)
#
# logits = tf.contrib.layers.fully_connected(logits1, 66177, tf.nn.relu)
# # Define a loss function
# loss = tf.reduce_sum( tf.square( y - logits ) )
# #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = logits))
#
# model = keras.Sequential([
#
#   keras.layers.Flatten(),
#   keras.layers.Dense(1000, activation=tf.nn.softmax, name='Softmax')
# ])
# # Define an optimizer
# train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#
# # Convert logits to label indexes
# # correct_pred = tf.argmax(logits, 1)
# #
# # # Define an accuracy metric
# # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# # print("images_flat: ", images_flat)
# # print("logits: ", logits)
# # print("loss: ", loss)
# # print("predicted_labels: ", correct_pred)
#
# tf.set_random_seed(1234)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
#
# sess.run(tf.global_variables_initializer())
#
# for i in range(2):
#     print('EPOCH', i)
#     _ = sess.run(train_op, feed_dict={x:[[[1.,0.],[1.,0.]],[[1.,0.],[1.,0.]]], y: [1,1]})
#     print("Loss: ", loss)
#     print('DONE WITH EPOCH')
# sess.close()


import numpy as np
import tensorflow as tf
from tensorflow import keras
def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(66177,))
    x1 = keras.layers.Dense(100, activation="relu", name="dense_1")(inputs)
    # x2 = keras.layers.Dense(40000, activation="relu", name="dense_2")(x1)
    outputs = keras.layers.Dense(66177)(x1)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


model = get_model()

# Train the model.
test_input = np.array([np.load("adinSpec.npy").flatten()])
print(test_input.shape)
test_target = np.array([np.load("artinSpec.npy").flatten()])
print(test_target.shape)
model.fit(test_input, test_target,10,1000)

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
