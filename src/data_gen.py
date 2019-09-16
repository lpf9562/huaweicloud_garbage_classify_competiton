# -*- coding: utf-8 -*-
import os
import math
import codecs
import random
import numpy as np
# from imgaug import augmenters as iaa
from glob import glob
from PIL import Image
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, Sequence
from sklearn.model_selection import train_test_split
from PIL import ImageEnhance

class BaseSequence(Sequence):
    """
    基础的数据流生成器，每次迭代返回一个batch
    BaseSequence可直接用于fit_generator的generator参数
    fit_generator会将BaseSequence再次封装为一个多进程的数据流生成器
    而且能保证在多进程下的一个epoch中不会重复取相同的样本
    """
    def __init__(self, img_paths, labels,labels_one_hot ,batch_size, img_size,tag=None):
        assert len(img_paths) == len(labels), "len(img_paths) must equal to len(lables)"
        assert img_size[0] == img_size[1], "img_size[0] must equal to img_size[1]"
        self.x_y = np.hstack((np.array(img_paths).reshape(len(img_paths), 1), np.array(labels_one_hot)))
        self.batch_size = batch_size
        self.img_size = img_size
        self.tag=tag
    def __len__(self):
        return math.ceil(len(self.x_y) / self.batch_size)

    @staticmethod
    def center_img(img, size=None, fill_value=255):
        """
        center img in a square background
        """
        h, w = img.shape[:2]
        if size is None:
            size = max(h, w)
        shape = (size, size) + img.shape[2:]
        background = np.full(shape, fill_value, np.uint8)
        center_x = (size - w) // 2
        center_y = (size - h) // 2
        background[center_y:center_y + h, center_x:center_x + w] = img
        return background

    def preprocess_img(self, img_path):
        """
        image preprocessing
        you can add your special preprocess method here
        """
        img = Image.open(img_path)

        resize_scale = self.img_size[0] / max(img.size[:2])
        img = img.resize((int(img.size[0] * resize_scale), int(img.size[1] * resize_scale)))
        img = img.convert('RGB')
        img = np.array(img)
        img = self.data_augment(img)
        img = img[:, :, ::-1]

        img = self.center_img(img, self.img_size[0])

        return img

    def __getitem__(self, idx):
        batch_x = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.x_y[idx * self.batch_size: (idx + 1) * self.batch_size, 1:]
        batch_x = np.array([self.preprocess_img(img_path) for img_path in batch_x])
        batch_y = np.array(batch_y).astype(np.float32)
        # for i in range(self.batch_size):
        #     batch_x[i]=self.data_augment(batch_x[i])

        datagen = ImageDataGenerator(rescale=1. / 255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')
        # i=0
        for batch, batch_yy in datagen.flow(batch_x, batch_y):
            # print("batch_x_shape:" + str(np.shape(batch)))
            # print("batch_y_shape:" + str(np.shape(batch_yy)))
            # for root, dirs, files in os.walk('E:\garbage_classify\garbage_classify\\test'):
            #     for file in files:
            #         name_split = file.split('.')
            #         last = '.txt'
            #         name = name_split[0] + last
            #         wname = os.path.join(root, name)
            #         if not os.path.exists(wname):
            #             file = open(wname, 'w')
            #             # str=name+','+labels[count]
            #             file.write(name)
            #             file.write(',')
            #             file.write(str(labels[j]))

            # print(batch_x.shape)
            # print(labels.shape)
            # batch_y=
            # batch_x=batch
            # batch_y=batch_yy
            # i += 1
            # if i > 0:
            #     break

        return batch_x, batch_y

    # 以下函数都是一些数据增强的函数
    def gamma_transform(self,img, gamma=1):

        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]

        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)

        return cv2.LUT(img, gamma_table)

    def random_gamma_transform(self,img, gamma_vari=2):
        log_gamma_vari = np.log(gamma_vari)

        alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)

        gamma = np.exp(alpha)

        return self.gamma_transform(img, gamma)

    def rotate(self,img,angle):
        size=224
        M_rotate = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)

        img = cv2.warpAffine(img, M_rotate, (size, size))

        # yb = cv2.warpAffine(yb, M_rotate, (size, size))

        return img

    def blur(self,img):
        img = cv2.blur(img, (3, 3))

        return img

    def add_noise(self,img):
        size=100
        for i in range(size):  # 添加点噪声

            temp_x = np.random.randint(0, img.shape[0])

            temp_y = np.random.randint(0, img.shape[1])

            img[temp_x][temp_y] = 255

        return img

    def data_augment(self,img):
        # if np.random.random() < 0.25:
        #     img = cv2.equalizeHist(img)
        # if np.random.random() < 0.25:
        #     img=cv2.normalize(img,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
        if np.random.random() < 0.25:
            img = self.rotate(img,90)

        if np.random.random() < 0.25:
            img = self.rotate(img, 180)

        if np.random.random() < 0.25:
            img = self.rotate(img, 270)

        if np.random.random() < 0.25:
            img = cv2.flip(img, 1)  # flipcode > 0：沿y轴翻转

        if np.random.random() < 0.25:
            img = self.random_gamma_transform(img, 1)

        if np.random.random() < 0.25:
            img = self.blur(img)

        # 双边过滤
        if np.random.random() < 0.25:
            img = cv2.bilateralFilter(img, 9, 75, 75)

        #  高斯滤波
        if np.random.random() < 0.25:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        if np.random.random() < 0.2:
            img = self.add_noise(img)

        return img
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.shuffle(self.x_y)


def data_flow(train_data_dir, batch_size, num_classes, input_size):  # need modify
    label_files = glob(os.path.join(train_data_dir, '*.txt'))
    random.shuffle(label_files)
    img_paths = []
    labels = []
    for index, file_path in enumerate(label_files):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            line = f.readline()
        line_split = line.strip().split(',')
        if len(line_split) != 2:
            print('%s contain error lable' % os.path.basename(file_path))
            continue
        img_name = line_split[0]
        label = int(line_split[1])
        img_paths.append(os.path.join(train_data_dir, img_name))
        labels.append(label)
    train_img_paths, validation_img_paths, train_labels, validation_labels = \
        train_test_split(img_paths, labels, test_size=0.25, random_state=0)
    train_labels_one_hot = np_utils.to_categorical(train_labels, num_classes)
    validation_labels_one_hot=np_utils.to_categorical(validation_labels, num_classes)
    print('total samples: %d, training samples: %d, validation samples: %d' % (len(img_paths), len(train_img_paths), len(validation_img_paths)))

    train_sequence = BaseSequence(train_img_paths, train_labels,train_labels_one_hot, batch_size, [input_size, input_size],'Train')
    validation_sequence = BaseSequence(validation_img_paths, validation_labels,validation_labels_one_hot, batch_size, [input_size, input_size],tag=None)
    # # 构造多进程的数据流生成器
    # train_enqueuer = OrderedEnqueuer(train_sequence, use_multiprocessing=True, shuffle=True)
    # validation_enqueuer = OrderedEnqueuer(validation_sequence, use_multiprocessing=True, shuffle=True)
    #
    # # 启动数据生成器
    # n_cpu = multiprocessing.cpu_count()
    # train_enqueuer.start(workers=int(n_cpu * 0.7), max_queue_size=10)
    # validation_enqueuer.start(workers=1, max_queue_size=10)
    # train_data_generator = train_enqueuer.get()
    # validation_data_generator = validation_enqueuer.get()

    # return train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator
    return train_sequence, validation_sequence


if __name__ == '__main__':
    # train_enqueuer, validation_enqueuer, train_data_generator, validation_data_generator = data_flow(dog_cat_data_path, batch_size)
    # for i in range(10):
    #     train_data_batch = next(train_data_generator)
    # train_enqueuer.stop()
    # validation_enqueuer.stop()
    train_sequence, validation_sequence = data_flow(train_data_dir, batch_size)
    batch_data, bacth_label = train_sequence.__getitem__(5)
    label_name = ['cat', 'dog']
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    train_sequence.on_epoch_end()
    batch_data, bacth_label = train_sequence.__getitem__(5)
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_2_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    train_sequence.on_epoch_end()
    batch_data, bacth_label = train_sequence.__getitem__(5)
    for index, data in enumerate(batch_data):
        img = Image.fromarray(data[:, :, ::-1])
        img.save('./debug/%d_3_%s.jpg' % (index, label_name[int(bacth_label[index][1])]))
    print('end')
