#-*- coding: utf-8 -*-
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
size = 224, 224
start = 0
end = 19600
"""
train 19600
train 19600*4
"""
for i in range(start,end):
    if i % 1000 == 0:
        print str(i)+" 번째 이미지를 처리 중입니다."
    image = Image.open("image/kalph_train"+str(i)+".jpg")
    image = image.resize(size)
    #image.thumbnail(size, Image.ANTIALIAS)
    np_array = np.array(image)
    #print np_array.shape
    plt.imsave("image_resize224/kalph_train" + str(i) + ".jpg", image, format='png', cmap='gray_r')
