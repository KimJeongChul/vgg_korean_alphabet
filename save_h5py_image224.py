#-*- coding: utf-8 -*-
import numpy as np
import h5py
from PIL import Image

start = 1
end = 19600
label_start = 19600
idx = 19600

with h5py.File('resized_kalph_train.hf', 'r') as hf:
    origin_images = np.array(hf['images'])
    labels = np.array(hf['labels'])
    print "original kalph_train images shape"+str(origin_images.shape)
    print "original kalph_train labels shape"+str(labels.shape)
    
image = Image.open("image_resize224/kalph_train0.jpg")
image = image.convert('L')
images = np.array(image)
images = images.reshape([1,224,224])

for i in range(start,end):
    if i % 1000 == 0 :
        print str(i)+"번째 image를 처리하였습니다"
    image = Image.open("image_resize224/kalph_train"+str(i)+".jpg")
    image = image.convert('L')
    np_array = np.array(image)
    np_array = np_array.reshape([1,224,224])
    images = np.append(images, np_array, axis=0)

for i in range(label_start, end):
    if (i-idx) % 5000 == 0:
        print str(i-idx)+"번째 label를 처리하였습니다."
    labels = np.append(labels, labels[(i % idx) - 1])


print "Add Data Augmentation"
print "After kalph_train images shape"+str(images.shape)
print "After kalph_train labels shape"+str(labels.shape)

h5f = h5py.File('kalph_224_train19600.hf','w')
print "h5py 파일을 생성합니다."
h5f.create_dataset('images', data=images)
h5f.create_dataset('labels', data=labels)
h5f.close()

with h5py.File('kalph_224_train19600.hf', 'r') as hf:
    images = np.array(hf['images'])
    labels = np.array(hf['labels'])
    print "kalph_augmentation_train images shape"+str(images.shape)
    print "kalph_augmentation_train labels shape"+str(labels.shape)
