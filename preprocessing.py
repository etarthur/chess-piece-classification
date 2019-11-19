import os
import cv2
import numpy as np
chess_piece_types = ['bishop', 'rook', 'pawn', 'knight']

'''
This preprocessing technique is based off of the section for generating additional data given a small dataset at 
https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8
'''


average = [0,0,0,0]
image_averages = []
count = 0
for dir in os.listdir("data/validation"):
        for image in os.listdir("data/validation/" + dir ):
            img = cv2.imread("data/validation/" + dir + "/"+image)
            channels = cv2.split(img)
            channels[0] = np.subtract(channels[0],105.74159353431077)
            channels[0] = np.divide(channels[0],49.08)
            channels[1] = np.subtract(channels[1], 110.12146428959164)
            channels[1] = np.divide(channels[1], 44.83)
            channels[2] = np.subtract(channels[2], 120.71599085974113)
            channels[2] = np.divide(channels[2], 38.25)
            new_img = cv2.merge(channels)
            cv2.imwrite("data/validation/" + dir + "/n_"+image,new_img)
            #temp = cv2.mean(img)
            #average[0]+=temp[0]
            #average[1]+=temp[1]
            #average[2]+=temp[2]
            #average[3]+=temp[3]
            #image_averages.append([temp[0],temp[1],temp[2]])
            #count +=1
exit(0)

avgr = average[0]/count
avgb = average[1]/count
avgg = average[2]/count


stdr = 0
stdb = 0
stdg = 0

for i in image_averages:
    stdr += (avgr-i[0])*(avgr-i[0])
    stdb += (avgb - i[1])*(avgb - i[1])
    stdg += (avgg - i[2])*(avgg - i[2])

stdr/=count-1
stdb/=count-1
stdg/=count-1

print("red average: ",avgr," std: ",stdr)
print("blue average: ",avgb," std: ",stdb)
print("green average: ",avgg," std: ",stdg)


