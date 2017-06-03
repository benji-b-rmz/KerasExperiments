# Multi-Layer NN for the CIFAR-10 Dataset
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# reshape to be [samples][pixels][width][height]
# reshape to single dim from RGB image of 32 * 32,
# 3 * 32 * 32 = 3072
print(X_train.shape[1:])
image_size = 3 * 32 * 32
X_train = X_train.reshape(X_train.shape[0], image_size).astype('float32')
X_test = X_test.reshape(X_test.shape[0], image_size).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


# define the larger model
def single_layer_model():
    # create model
    model = Sequential()
    # convolution -> pooling -> dropout
    model.add(Dense(2048, input_shape=(image_size,), activation='relu'))
    model.add(Dense(2048, input_shape=(image_size,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, input_shape=(image_size,), activation='relu'))
    model.add(Dense(1024, input_shape=(image_size,), activation='relu'))
    model.add(Dense(512, input_shape=(image_size,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = single_layer_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Multi-Layer NN Error: %.2f%%" % (100 - scores[1] * 100))
