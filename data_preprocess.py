from mtcnn import MTCNN
import numpy as np
from PIL import Image
from os import listdir
import matplotlib.pyplot as plt
import os
from skimage import io
import random

def cut_figure(model, file_dir, size):
    img = Image.open(file_dir).convert('RGB')
    pix = np.asarray(img)
    face = model.detect_faces(pix)
    if len(face) == 0:
        return None
    res = []
    for i in face:
        start_x, start_y, width, height = i['box']
        end_x = abs(start_x) + width
        end_y = abs(start_y) + height
        face_pix = pix[start_y:end_y, start_x:end_x]
        img = Image.fromarray(face_pix).resize(size)
        face_arr = np.asarray(img)
        res.append(face_arr)
    res = np.asarray(res)
    return res


class data_loader:
    def __init__(self):
        self.model = MTCNN()

    def cut_figure(self, file_dir, size):
        img = Image.open(file_dir).convert('RGB')
        pix = np.asarray(img)
        face = self.model.detect_faces(pix)
        if len(face) == 0:
            return None
        res = []
        for i in face:
            start_x, start_y, width, height = i['box']
            end_x = abs(start_x) + width
            end_y = abs(start_y) + height
            face_pix = pix[start_y:end_y, start_x:end_x]
            img = Image.fromarray(face_pix).resize(size)
            face_arr = np.asarray(img)
            res.append(face_arr)
        res = np.asarray(res)
        return res

    def fit(self, name, n=None, size=(80, 80)):
        res = []
        dir = os.getcwd()
        if name == 'cuhksz':
            dir = dir + "/cuhksz/"
        if name == 'asian':
            dir = dir + "/asian/"

        for files in listdir(dir):
            face = self.cut_figure(dir + files, size=size)
            if face is None:
                continue
            if len(face) > 1:
                for i in face:
                    res.append(i)
            else:
                res.append(face[0])
            if n is not None:
                if len(res) > n:
                    break
        return np.array(res)

    def transform(self, data, file_name):
        dir = os.getcwd()
        file_path = dir + '/' + file_name + '.npz'
        np.lib.npyio.savez_compressed(file_name + '.npz', data)
        return file_path

    def fit_transform(self, name, n=None, size=(80, 80)):
        data = self.fit(name, n, size)
        file_name = name + '_processed'
        res_path = self.transform(data, file_name)
        return res_path


def random_filename():
    lenth = np.random.randint(1, 26, size=1)
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', \
                'h', 'i', 'j', 'k', 'l', 'm', 'n', \
                'o', 'p', 'q', 'r', 's', 't', 'u', \
                'v', 'w', 'x', 'y', 'z']
    name = random.sample(alphabet, lenth[0])
    return ''.join(name)


def save_images(dir, filename):
    current_dir = os.getcwd()
    path = current_dir + filename
    os.makedirs(path)
    data = np.load(dir)
    data = data['arr_0']
    for i in data:
        img = Image.fromarray(i)
        name = random_filename()
        img.save(path + '/%s.jpg' % (name))

def main():
    os.chdir('/content/drive/MyDrive/Colab Notebooks/')
    loader = data_loader()
    cuhk_res = loader.fit('cuhksz')
    asian_res = loader.fit('asian')
    conb = np.vstack((cuhk_res,asian_res))
    res_path = loader.transform(conb, 'training_data')
    return res_path