# Benjamin Ramirez Jun 3, 2017
# loading saved mnist model and evaluating
import numpy as np
import os
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K

K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# again, this section is taken from Jason Brownlee's tutorial on saving/loading keras models
# http://machinelearningmastery.com/save-load-keras-deep-learning-models/
# loading json and creating model from saved files
json_file = open('./model_saves/mnist_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('./model_saves/mnist_cnn_model.h5')
print("Loaded model from disk")

# evaluate model and compare results to trained model:
# Expected Print:
# Prediction for X[0]: 7
# Large CNN Error: 0.77%
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

test_input_image = X_test[0]
print(test_input_image.shape)
output_test = loaded_model.predict(np.expand_dims(test_input_image, axis=0))
print("Prediction for X[0]:", np.argmax(output_test))

scores = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

