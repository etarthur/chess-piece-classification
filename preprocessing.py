import os

from canny import *

'''
This preprocessing technique is based off of the section for generating additional data given a small dataset at 
https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8
'''

folder = 'archive'  # Change to preprocess another location, this overwrites the images so save them first
for dir in os.listdir('data/' + folder):
    subdir = os.path.join('data', folder, dir)
    print(subdir)
    images = []
    names = []
    for image in os.listdir(subdir):
        if len(images) == 3:
            cv2.imwrite(os.path.join(subdir, names[0]), images[0])
            cv2.imwrite(os.path.join(subdir, names[1]), images[1])
            cv2.imwrite(os.path.join('data', 'validation', dir, names[2]), images[2])
            images = []
            names = []
        pic = cv2.imread(os.path.join(subdir, image), cv2.IMREAD_COLOR)
        pic = cv2.resize(pic, (300, 300))

        # Applying preprocessing methods
        blur_img = cv2.medianBlur(pic, 7)
        proc_img = canny(blur_img)
        # thr = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 1)

        images.append(proc_img)
        names.append(image)

    # Write the rest of the images
    for i in range(len(images)):
        dst = os.path.join('data', 'train', dir, names[i])
        cv2.imwrite(dst, images[i])
