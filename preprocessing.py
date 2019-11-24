import os
import random

import cv2

from canny import apply_canny_filter

chess_piece_types = ['bishop', 'rook', 'pawn', 'knight']

'''
This preprocessing technique is based off of the section for generating additional data given a small dataset at 
https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8
'''

average = [0, 0, 0, 0]
image_averages = []
count = 0
archive_data_root = 'data/archive'
training_data_root = 'data/train'
validation_data_root = 'data/validation'


def select_ten_percent(imlist):
    '''
    Randomly shuffles the image list and returns a training images list and validation
    images list that have a 9:1 distribution, respectively of the length of the original
    dataset.

    :param imlist: Original list of images to shuffle and distribute
    :return: Training image list and validation image list as a tuple
    '''
    n = int(len(imlist) * 0.15)
    random.shuffle(imlist)
    train_imlist = imlist[:n]
    validation_imlist = imlist[n:]
    assert len(imlist) is len(train_imlist) + len(validation_imlist)
    return train_imlist, validation_imlist


def delete_imgs_in_subdirs(dir):
    '''
    Deletes jpg/jpeg images in the given subdir.

    :param dir: Directory to scan subdirs for.
    '''
    for dir in os.listdir(dir):
        subdir = os.path.join(training_data_root, dir)
        flist = os.listdir(subdir)
        for f in flist:
            if f.endswith('.jpg') or f.endswith('.jpeg'):
                os.remove(os.path.join(subdir, f))


if __name__ == '__main__':
    # First clean out the training and validation data directories
    delete_imgs_in_subdirs(training_data_root)
    delete_imgs_in_subdirs(validation_data_root)

    # Then apply preprocessing on archive data and save to respective directories
    for dir in os.listdir(archive_data_root):
        subdir = os.path.join(archive_data_root, dir)
        imlist = os.listdir(subdir)
        train_imlist, validation_imlist = select_ten_percent(imlist)
        for image in train_imlist:
            impath = os.path.join(subdir, image)
            img = cv2.imread(impath)
            # TODO: Apply preprocessing methods
            # new_img = apply_canny_filter(img)
            cv2.imwrite(os.path.join(training_data_root, dir, image), img)
        for image in validation_imlist:
            impath = os.path.join(subdir, image)
            img = cv2.imread(impath)
            # TODO: Apply preprocessing methods
            # new_img = apply_canny_filter(img)
            cv2.imwrite(os.path.join(validation_data_root, dir, image), img)
    exit(0)

    for image in os.listdir(subdir):
        impath = os.path.join(subdir, image)
        img = cv2.imread(impath)
        # channels = cv2.split(img)
        # channels[0] = np.subtract(channels[0], 105.74159353431077)
        # channels[0] = np.divide(channels[0], 49.08)
        # channels[1] = np.subtract(channels[1], 110.12146428959164)
        # channels[1] = np.divide(channels[1], 44.83)
        # channels[2] = np.subtract(channels[2], 120.71599085974113)
        # channels[2] = np.divide(channels[2], 38.25)
        # new_img = cv2.merge(channels)
        new_img = apply_canny_filter(img)
        cv2.imwrite("data/canny_validation/" + dir + "/" + image, new_img)
        # temp = cv2.mean(img)
        # average[0]+=temp[0]
        # average[1]+=temp[1]
        # average[2]+=temp[2]
        # average[3]+=temp[3]
        # image_averages.append([temp[0],temp[1],temp[2]])
        # count +=1
        exit(0)

        avgr = average[0] / count
        avgb = average[1] / count
        avgg = average[2] / count

        stdr = 0
        stdb = 0
        stdg = 0

        for i in image_averages:
            stdr += (avgr - i[0]) * (avgr - i[0])
            stdb += (avgb - i[1]) * (avgb - i[1])
            stdg += (avgg - i[2]) * (avgg - i[2])

        stdr /= count - 1
        stdb /= count - 1
        stdg /= count - 1

        print("red average: ", avgr, " std: ", stdr)
        print("blue average: ", avgb, " std: ", stdb)
        print("green average: ", avgg, " std: ", stdg)
