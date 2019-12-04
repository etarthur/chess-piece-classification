import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

'''
Model instance class with training and testing methods. Classifies chess pieces in images from 
data/train/<type> where type is one of the classes initialized globally in this class.

Parameters and model structure example:
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
Image preprocessing techniques for a small dataset from:
https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8
Image classification techniques in Keras:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

For GPU support, tensorflow>=1.15.* must be installed, otherwise GPU packages can be installed with 
`pip install tensorflow-gpu==1.15`. The package versions included in requirements.txt include versions of
TensorFlow which are required to train the model on a GPU. If training is instead done without a CUDA-enabled
card, only the indicated version of Keras is required. Below are the details of how to enable GPU training.

This model builds on CUDA enabled NVIDIA GPUs with the following software requirements:
— NVIDIA® GPU drivers 
    — CUDA 10.0 requires 410.x or higher.
— CUDA® Toolkit —TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0)
    — CUPTI ships with the CUDA Toolkit.
— cuDNN SDK (>= 7.4.1)

Furthermore, the visualization method requires matplotlib.pyplot which uses GraphViz 3.8 to generate and
render the summary accuracy and loss graphs. GraphViz must be in PATH, otherwise it will not be recognized by 
matplotlib.pyplot. This is highly suggested to view the status of the model and identify overfitting, if 
applicable.
'''

classes = ['bishop', 'pawn', 'knight', 'rook']
num_classes = len(classes)


class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', input_shape=(300, 300, 3), activation='relu'))
        self.model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.1))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
        plot_model(self.model, to_file='model.png')
        self.model.summary()

    def train(self, batch_size=64, epochs=720):
        """
        Trains the model. Keras' ImageDataGenerators are used to flow data from directories
        data/train and data/validation where the subdirectory names are the data labels.
        """
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            rotation_range=180,
            height_shift_range=0.1,
            width_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)

        test_datagen = ImageDataGenerator()

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
            steps_per_epoch=1543 / batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=314 / batch_size)

        self.visualize(epochs, fit_generator)
        self.__save_weights('%s_epochs_model_weights.h5' % epochs)

    @staticmethod
    def visualize(epochs, fit_generator):
        """
        Plots all the applicable data from the training history.
        """
        for key in fit_generator.history.keys():
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(np.arange(0, epochs), fit_generator.history[key], label=key)
            plt.title(key)
            plt.xlabel('Epoch #')
            plt.ylabel(key)
            plt.legend(loc='lower left')
            plt.savefig(os.path.join('plots', '{}_model_{}_plot.png'.format(epochs, key)))

    def test(self, weights_file, img, classification):
        """
        Evaluates the given image with the given weights.
        """
        if self.model.weights is None:
            self.__load_weights(weights_file)
        score = self.model.evaluate()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def __save_weights(self, weights_file):
        """
        Saves the model weights to the provided file.
        """
        self.model.save_weights(os.path.join('weights', weights_file))

    def __load_weights(self, weights_file):
        """
        Loads the weights file provided.
        """
        self.model.load_weights(os.path.join('weights', weights_file))
