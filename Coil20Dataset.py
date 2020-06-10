from mxnet.gluon.data.dataset import Dataset
from mxnet.image import image as img
import os
import cv2 as cv
import numpy as np
class Coil20(Dataset):

    def __init__(self, data_path='train'):
        self.path = data_path
        directory = os.listdir(self.path)
        self.images = []
        for image in directory:

            label = image.split('_')[0]
            label = label[3:]
            image_dict = {'path': self.path + "/" + image, 'label': label}
            self.images.append(image_dict)
        self.len = len(self.images)
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = image_info['path']
        image = cv.imread(image_path)
        image = np.reshape(image,(3, image.shape[0], image.shape[1]))
        label = image_info['label']
        image = image.astype('float32')
        return image, float(label) -1
