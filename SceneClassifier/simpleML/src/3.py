import numpy as np
import pandas as pd
from PIL import Image
import os, sys

y_train = np.genfromtxt('data/train_labels.csv', delimiter=',')
y_test = np.genfromtxt('data/test_labels.csv', delimiter=',')

folder = 'data/train'
'''
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

print("Working with {0} images".format(len(onlyfiles)))
print("Image examples: ")
'''
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = list(range(1, 1889))
LEN = len(train_files)
print(LEN)
#print(list(train_files))

for i in range(1, 1889):
    #print(i)
    train_files[i-1]=str(train_files[i-1]) + '.jpg'

print(train_files[0])
print("Files in train_files: %d" % len(train_files))
y_train = np.array(y_train)
# Original Dimensions
image_width = 32
image_height = 32
channels = 3

dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)
i = 0
for _file in train_files:
    img = load_img(folder + "/" + str(_file))
    img = img.resize((32, 32))
    x = img_to_array(img)
    x = x.reshape((32, 32, 3))
    dataset[i] = x
    i += 1

# test data
folder = 'data/test'
onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

print("Working with {0} images".format(len(onlyfiles)))
print("Image examples: ")
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = list(range(1, 801))
LEN = len(train_files)

for i in range(1, 801):
    train_files[i-1] = str(train_files[i-1]) + '.jpg'

print("Files in train_files: %d" % len(train_files))
y_test = np.array(y_test)
# Original Dimensions
image_width = 32
image_height = 32
channels = 3

x_test = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                    dtype=np.float32)
i = 0
for _file in train_files:
    img = load_img(folder + "/" + _file)
    img = img.resize((32, 32))
    x = img_to_array(img)
    x = x.reshape((32, 32, 3))
    x_test[i] = x
    i += 1

from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

import keras
from keras.models import Model


def VGF16(img_shape, num_classes):
    model_16 = keras.applications.VGG16(include_top=False, weights='imagenet')
    keras_input = keras.layers.Input(shape=img_shape)
    output_vgg = model_16(keras_input)
    output_vgg = keras.layers.Flatten()(output_vgg)
    # x = keras.layers.Dense(2056, activation='relu')(output_vgg)
    # x = keras.layers.Dense(1028, activation='relu')(output_vgg)
    x = keras.layers.Dense(num_classes, activation='softmax')(output_vgg)
    # x = keras.layers.Dense(num_classes, activation='softmax')(output_vgg)
    model = Model(inputs=keras_input, outputs=x)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print
    model.layers[-1]
    # print (model.layers)
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-1:]:
        layer.trainable = True
    # model.layers[-1].trainable = False
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = VGF16((32, 32, 3), 9)
# model = VGF16((32,32,3), 10)
# model.summary()

# x_train = x_train[0:100,:,:,:]
# y_train = y_train[0:100,:]
# callback = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1, write_grads=True, write_graph=True,write_images=True)

hist = model.fit(dataset, y_train, epochs=10)
# pred = model.predict(x_train)
# print model.evaluate(dataset, y_train)
print(model.evaluate(x_test, y_test))
