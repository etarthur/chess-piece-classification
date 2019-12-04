import sys

import cv2

from canny import *
from model import *

if __name__ == '__main__':
    weights_file = sys.argv[1]
    images = sys.argv[2:]
    m = Model()
    for path in images:
        # classification = # (img.split('/')[-1]).split('.')[0]
        # print(classification)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError('No image loaded given path {}'.format(path))
        blur_img = cv2.medianBlur(img, 7)
        proc_img = canny(blur_img)
        prediction = m.predict(weights_file, proc_img)
        print(prediction)
