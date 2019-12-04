import cv2


def adaptive_threshold(impath):
    """
    Applies adaptive thresholding on the image path provided.
    Returns a processed image matrix.
    """
    img = cv2.resize(cv2.imread(impath, cv2.IMREAD_GRAYSCALE), (300, 300), interpolation=cv2.INTER_AREA)
    proc_img = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.adaptiveThreshold(proc_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
