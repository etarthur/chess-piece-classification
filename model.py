import os
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.applications.resnet50 import ResNet50

# r = ResNet50()

'''
Model instance class with training and testing methods. Classifies chess pieces in images from 
data/train/<type> where type is one of chess_piece_types.

Parameters and model structure example:
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
Image preprocessing techniques for a small dataset from:
https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8
Image classification techniques in Keras:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

For GPU support, tensorflow>=1.15.* must be installed, otherwise GPU packages can be installed with 
`pip install tensorflow-gpu==1.15`

This model builds on CUDA enabled NVIDIA GPUs with the following software requirements:
- NVIDIA® GPU drivers —CUDA 10.0 requires 410.x or higher.
- CUDA® Toolkit —TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0)
    - CUPTI ships with the CUDA Toolkit.
- cuDNN SDK (>= 7.4.1)
'''

chess_piece_types = ['bishop', 'rook', 'pawn', 'knight']
num_classes = len(chess_piece_types)


def visualize(epochs, fit_generator):
    acc = fit_generator.history['accuracy']
    val_acc = fit_generator.history['val_accuracy']

    loss = fit_generator.history['loss']
    val_loss = fit_generator.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # self.model.add(Dropout(0.5))

        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['categorical_accuracy'])
        self.model.summary()

    def train(self, batch_size=16, epochs=10):
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            rotation_range=180,
            height_shift_range=0.1,
            width_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[.2, .1])

        test_datagen = ImageDataGenerator(
            rotation_range=180,
            height_shift_range=0.1,
            width_shift_range=0.1,
            brightness_range=[.2, .1])

        train_generator = train_datagen.flow_from_directory(
            'data\\train',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

        validation_generator = test_datagen.flow_from_directory(
            'data\\validation',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

        fit_generator = self.model.fit_generator(
            train_generator,
            steps_per_epoch=300,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=78)

        # visualize(epochs, fit_generator)
        self.__save_weights('%s_epochs_gang_gang.h5' % epochs)

    def test(self, weights_file, img, classification):
        if self.model.weights is None:
            self.__load_weights(weights_file)
        score = self.model.evaluate()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def __save_weights(self, weights_file):
        self.model.save_weights(os.path.join('weights', weights_file))

    def __load_weights(self, weights_file):
        self.model.load_weights(os.path.join('weights', weights_file))
