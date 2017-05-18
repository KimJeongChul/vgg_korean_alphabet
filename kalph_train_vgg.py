#-*- coding: utf-8 -*-
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import dtypes
import random

"""
kalph_train.hf (train_data 19,600자)
image 52*52
kalph_test.hf (test_data 3,920자)

클래스
가0 나1 다2 라3 마4 바5 사6 아7 자8
차9 카10 파11 타12 하13

VGG MNIST로 학습시킨 값 weight를 가져와
VGG로 다시 재학습을 시켜보자

"""

class DataSet(object):

    def __init__(self, images, labels, test_images, test_labels, dtype=dtypes.float32, reshape=True):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = 19600
        self._num_classes = 14
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            test_images = test_images.reshape(test_images.shape[0],
                                              test_images.shape[1] * test_images.shape[2])
        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
            test_images = test_images.astype(np.float32)
            test_images = np.multiply(test_images, 1.0 / 255.0)

        # dense_to_one_hot
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * self._num_classes
        labels_one_hot = np.zeros((num_labels, self._num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1

        num_test_labels = test_labels.shape[0]
        test_index_offset = np.arange(num_test_labels) * self._num_classes
        test_labels_one_hot = np.zeros((num_test_labels, self._num_classes))
        test_labels_one_hot.flat[test_index_offset + test_labels.ravel()] = 1

        self._images = images
        self._labels = labels_one_hot

        self._test_images = test_images
        self._test_labels = test_labels_one_hot

        print self._images.shape
        print self._labels.shape

    # Data Augmentation
    def distorted_inputs(self, image):
        # Randomly crop a [height, width] section of the image.
        #height = width = 40
        #distorted_image = tf.random_crop(images, [height, width, 3])
        # Randomly flip the image horizontally.
        #distorted_image = tf.image.random_flip_left_right(images)
        #degrees = random.randint(0,360)
        #image = Image.Image.rotate(image, degrees)
        image = np.rot90(image,1)
        distorted_image = tf.image.random_brightness(image, max_delta=63)
        #distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        return distorted_image

    def get_data_sets(self):
        return images, labels, test_images, test_labels

    def train_next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > num_imgs:
            self._epoch_completed += 1

        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)

        self._images = self._images[perm]
        self._labels = self._labels[perm]

        start = 0
        self._index_in_epoch = batch_size
        end = self._index_in_epoch


        return self._images[start:end], self._labels[start:end]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':
    with tf.device('gpu:0'):
        class_hangul = ['가', '나', '다', '라', '마', '바', '사', '아', '자', '차', '카', '파', '타', '하']
        with h5py.File('kalph_224_train19600.hf', 'r') as hf:
            images = np.array(hf['images'])
            labels = np.array(hf['labels'])
        with h5py.File('kalph_224_test.hf', 'r') as hf:
            test_images = np.array(hf['images'])
            test_labels = np.array(hf['labels'])

        num_imgs, rows, cols = images.shape

        data_set = DataSet(images, labels, test_images, test_labels)
        images, labels, test_images, test_labels = data_set.get_data_sets()

        #plt.show()
        x_input = tf.placeholder(tf.float32, [None, 224*224])
        y_input = tf.placeholder(tf.float32, [None, 14])

        # 1st Convolutional Layer 224*224*64
        W_conv1_1 = weight_variable([3, 3, 1, 64])
        b_conv1_1 = bias_variable([64])
        x_image = tf.reshape(x_input, [-1,224,224,1])

        h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)
    
        W_conv1_2 = weight_variable([3, 3, 64, 64])
        b_conv1_2 = bias_variable([64])
        h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)

        h_pool1 = max_pool_2x2(h_conv1_2)

        # 2nd Convolutional Layer 112*112*128
        W_conv2_1 = weight_variable([3, 3, 64, 128])
        b_conv2_1 = bias_variable([128])
        h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)

        W_conv2_2 = weight_variable([3, 3, 128, 128])
        b_conv2_2 = bias_variable([128])
        h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)    

        h_pool2 = max_pool_2x2(h_conv2_2)

        # 3nd Convolutional Layer 56*56*256
        W_conv3_1 = weight_variable([3, 3, 128, 256])
        b_conv3_1 = bias_variable([256])
        h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1)

        W_conv3_2 = weight_variable([3, 3, 256, 256])
        b_conv3_2 = bias_variable([256])
        h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2)

        W_conv3_3 = weight_variable([3, 3, 256, 256])
        b_conv3_3 = bias_variable([256])
        h_conv3_3 = tf.nn.relu(conv2d(h_conv3_2, W_conv3_3) + b_conv3_3)

        h_pool3 = max_pool_2x2(h_conv3_3)

        # 4nd Convolutional Layer 28*28*512
        W_conv4_1 = weight_variable([5, 5, 256, 512])
        b_conv4_1 = bias_variable([512])
        h_conv4_1 = tf.nn.relu(conv2d(h_pool3, W_conv4_1) + b_conv4_1)

        W_conv4_2 = weight_variable([5, 5, 512, 512])
        b_conv4_2 = bias_variable([512])
        h_conv4_2 = tf.nn.relu(conv2d(h_conv4_1, W_conv4_2) + b_conv4_2)

        W_conv4_3 = weight_variable([5, 5, 512, 512])
        b_conv4_3 = bias_variable([512])
        h_conv4_3 = tf.nn.relu(conv2d(h_conv4_2, W_conv4_3) + b_conv4_3)

        h_pool4 = max_pool_2x2(h_conv4_3)

        # 5nd Convolutional Layer 14*14*512
        W_conv5_1 = weight_variable([5, 5, 512, 512])
        b_conv5_1 = bias_variable([512])
        h_conv5_1 = tf.nn.relu(conv2d(h_pool4, W_conv5_1) + b_conv5_1)

        W_conv5_2 = weight_variable([5, 5, 512, 512])
        b_conv5_2 = bias_variable([512])
        h_conv5_2 = tf.nn.relu(conv2d(h_conv5_1, W_conv5_2) + b_conv5_2)

        W_conv5_3 = weight_variable([5, 5, 512, 512])
        b_conv5_3 = bias_variable([512])
        h_conv5_3 = tf.nn.relu(conv2d(h_conv5_2, W_conv5_3) + b_conv5_3)

        h_pool5 = max_pool_2x2(h_conv5_3)

        # 1st Fully Connected Layer
        shape = int(np.prod(h_pool5.get_shape()[1:]))
        W_fc1 = weight_variable([shape, 4096])
        b_fc1 = bias_variable([4096])

        h_pool5_flat = tf.reshape(h_pool5, [-1, shape])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

        # Drop out
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 2nd Fully Connected Layer
        W_fc2 = weight_variable([4096, 14])
        b_fc2 = bias_variable([14])
        y_conv = tf.matmul(h_fc1_drop, W_fc2)

        cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_input,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for i in range(3000):
                batch_xs, batch_ys = data_set.train_next_batch(50)

                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x_input: batch_xs, y_input: batch_ys, keep_prob: 1.0})
                    print "step : "+str(i)+" train accuracy : " + str(train_accuracy)
                train_step.run(feed_dict={x_input: batch_xs, y_input: batch_ys, keep_prob: 0.5})

            test_accuracy = accuracy.eval(
            feed_dict={x_input: test_images, y_input: test_labels, keep_prob: 1.0})
            print "test accuracy : " + str(test_accuracy)
