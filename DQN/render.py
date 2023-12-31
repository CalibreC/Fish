#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/8 12:38
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : render.py
# @Description      :
import cv2
import numpy as np
from matplotlib import pyplot as plt


class PltRender:
    def __init__(self, call_back=None):
        self.fig = plt.figure(figsize=(6, 1.5))
        plt.ion()
        plt.tight_layout()
        # plt.clf()
        self.call_back = call_back

    def draw(self, low, high, pointer, ticks):
        w, h = 300, 50
        img = np.zeros((h, w, 3), np.uint8)
        img[:, int(low * w) - 3 : int(low * w) + 3, :] = np.array([255, 0, 0])
        img[:, int(pointer * w) - 3 : int(pointer * w) + 3, :] = np.array([0, 255, 0])
        img[:, int(high * w) - 3 : int(high * w) + 3, :] = np.array([0, 0, 255])

        cv2.imshow("img", img)
        cv2.waitKey(1)
        # plt.imshow(img)
        # plt.title(f'tick:{ticks}')
        # plt.draw()
        if self.call_back:
            self.call_back()
        plt.pause(0.0001)
        # plt.clf()
