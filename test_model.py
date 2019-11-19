from model import *
import cv2
import numpy as np

if __name__ == '__main__':
    # TODO: Available GPU check for keras & tf backend
    # from keras import backend
    # if len(backend.tensorflow_backend._get_available_gpus()) == 0:
    #     print('WARNING: No available GPUs found by Keras')
    m = Model()
    m.train()
    exit(0)
    m.model.load_weights("weights/10_epochs_gang.h5")
    #ximg = load_img("data/validation/bishop/20191108_142519.jpg",target_size=(300,300,3))
    img = cv2.imread("data/train/pawn/IMG_2166.jpg")
    img = cv2.resize(img,(300,300))
    average = cv2.mean(img)
    avg = np.sum(average)
    std = [0,0,0]
    for i in range(3):
        std[i] = np.sqrt((average[i] - avg) * (average[i]-avg))
    channels = cv2.split(img)
    channels[0] = np.subtract(average[0],channels[0])
    channels[0] = np.divide(channels[0], std[0])
    channels[1] = np.subtract(average[1],channels[1])
    channels[0] = np.divide(channels[0], std[1])
    channels[2] = np.subtract(average[2],channels[2])
    channels[0] = np.divide(channels[0], std[2])
    ximg = cv2.merge(channels)
    x = ximg.reshape((1,) + ximg.shape)
    print(m.model.predict(x))