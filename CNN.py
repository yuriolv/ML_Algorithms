import tensorflow as tf
from tensorflow import keras
from keras.src.datasets import cifar10
from keras.src.models import Sequential 
from keras.src.layers import Dense, Dropout, Activation, Flatten
from keras.src.layers import Conv2D, MaxPooling2D



dataset = cifar10
((x_train, y_train), (x_test, y_test)) = dataset.load_data()

x_train = x_train/255.0
x_test = x_test/255.0
""" y_train = y_train/255.0
y_test = y_test/255.0 """

x_train = x_train.reshape((x_train.shape[0], 32,32,3))
x_test = x_test.reshape((x_test.shape[0], 32,32,3))

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('sigmoid'))

model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.2)
