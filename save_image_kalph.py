#-*- coding: utf-8 -*-

import h5py
import numpy as np
import matplotlib.pyplot as plt



with h5py.File('resized_kalph_train.hf', 'r') as hf:
    images = np.array(hf['images'])

start = 0
end = 19600
for i in range(start,end):
    if i % 1000 == 0:
        print str(i)+"번째 image를 처리하였습니다"
    plt.imsave("image/kalph_train"+str(i)+".jpg", images[i], format='png', cmap='gray_r')
