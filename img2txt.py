#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

"""
# @Time    : 20-7-9 下午5:21

# @Author  : zhufa

# @Software: PyCharm
"""
"""
将图像转换为28×28的尺寸后，存储为txt格式，供minist_test调用
"""
import os
from PIL import Image
import cv2 as cv
import math
import numpy as np


def resizeIMG(img, size):
    w, h = img.shape
    s = max(w, h)
    tmpsize = int(math.ceil(s / size) * size)
    tmpimg = np.zeros((tmpsize, tmpsize), dtype=np.uint8)
    start_x = (tmpsize - w) / 2
    start_y = (tmpsize - h) / 2
    tmpimg[start_x:start_x + w, start_y:start_y + h] = img
    return tmpimg


def init(fileList, n):
    for i in range(0, n):
        img = Image.open("test/img/" + str(fileList[i]))
        new_img = img.resize((28, 28))
        img1 = cv.imread("test/img/" + str(fileList[i]), cv.IMREAD_GRAYSCALE)
        img2 = resizeIMG(img1, 28.0)
        img2 = cv.resize(img2, (28, 28))
        cv.imwrite("test/img1/" + str(fileList[i]), img2)
        _, binary = cv.threshold(img2, 127, 255, cv.THRESH_BINARY)

        w, h = binary.shape
        with open("test/txt/" + str(fileList[i]).split('.')[0] + ".txt", "w") as f:
            for c in range(h):
                for j in range(w):
                    f.write(str(binary[c][j] / 255) + " ")
                    if j == w - 1:
                        f.write("\n")
        f.close()


if __name__ == "__main__":
    fileList = os.listdir("test/img/")
    n = len(fileList)
    init(fileList, n)
