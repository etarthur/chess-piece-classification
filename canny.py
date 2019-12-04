import cv2
import numpy as np


def __auto_canny(img, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(img)

    # Apply Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    return edged


def canny(img):
    # img = cv2.imread(impath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    auto = __auto_canny(blurred)
    return auto
