#-*- coding: utf-8 -*-
import random
from PIL import Image
import matplotlib.pyplot as plt
start = 0
end = 19600
"""
train 19600
train 19600*4
"""
for i in range(start,end):
    image = Image.open("image_resize224/kalph_train"+str(i)+".jpg")
    degrees = random.randint(-20,20)
    if i % 1000 == 0:
        print str(i)+"번째 image를 처리하였습니다"
    image = Image.Image.rotate(image, degrees)
    plt.imsave("image_augmentation224/kalph_train" + str(i+end) + ".jpg", image, format='png', cmap='gray_r')

    degrees = random.randint(-20, 20)
    image = Image.Image.rotate(image, degrees)
    plt.imsave("image_augmentation224/kalph_train" + str(i+(end * 2)) + ".jpg", image, format='png', cmap='gray_r')

    degrees = random.randint(-20, 20)
    image = Image.Image.rotate(image, degrees)
    plt.imsave("image_augmentation224/kalph_train" + str(i+(end * 3)) + ".jpg", image, format='png', cmap='gray_r')

    degrees = random.randint(-20, 20)
    image = Image.Image.rotate(image, degrees)
    plt.imsave("image_augmentation224/kalph_train" + str(i+(end * 4)) + ".jpg", image, format='png', cmap='gray_r')

print "Data Augmentation 개수 : " + str(end*4)
print "Train Data 개수 : " + str(end*5)
