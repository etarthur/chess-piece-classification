import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator,load_img,img_to_array
from keras.optimizers import Adadelta
from keras.constraints import max_norm
from keras.losses import categorical_crossentropy

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


class Model:
    def __init__(self):
        self.model = Sequential()
        # TODO: Try with two 32-filter convolution + pooling layer iterations
        self.model.add(Conv2D(32,
                               (3, 3),
                              input_shape=(300, 300, 3),padding='same',
                                activation='relu'))
        self.model.add(Conv2D(32,
                              (3, 3),padding='same',
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64,
                              (3, 3),padding='same',
                              activation='relu'))
        self.model.add(Conv2D(64,
                              (3, 3), padding='same',
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(num_classes, activation='softmax'))

        # TODO: Evaluate other optimizers
        self.model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

    def train(self, batch_size=16, epochs=20):
        train_datagen = ImageDataGenerator(
            shear_range=0.2,
            rotation_range=180,
            height_shift_range=0.1,
            width_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[.2,.1])

        test_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True)

        validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(300, 300),
            batch_size=batch_size,
            class_mode='categorical')


        self.model.fit_generator(
            train_generator,
            steps_per_epoch=351,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=46)

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
