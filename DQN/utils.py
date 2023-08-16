#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/13 1:33
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : utils.py
# @Description      :
import cv2
import numpy as np


# 获取有效区域
def get_valid_region(full_image=None, region=None):
    bar_template = cv2.imread("./imgs/bar_template.png")
    bar_y_pos = find_bar_y_pos(full_image, bar_template)
    left = 712 - 10
    top = bar_y_pos
    width = 496 + 20
    height = 103
    # print(left, top, width, height)
    valid_image = full_image[top : top + height, left : left + width]

    return valid_image


def match_image(image, target, method=cv2.TM_CCOEFF):
    """
    https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    """
    h, w = target.shape[:2]
    res = cv2.matchTemplate(image, target, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return (
        *top_left,
        *bottom_right,
        top_left[0] + w // 2,  # center
        top_left[1] + h // 2,
    )


def find_exit(img=None, exit_template=None):
    # 根据退出标志位置，定位画面范围
    exit_pos = match_image(img, exit_template)
    return exit_pos


def find_bar_y_pos(img=None, bar_template=None):
    """
    模板匹配位置有偏差，只取有效的纵坐标，方便后续截图
    """
    bar_pos = match_image(img, bar_template)
    return bar_pos[1] - 9


if __name__ == "__main__":
    full_image = cv2.imread("../imgs/test/test2.jpg")
    image = get_valid_region(full_image, None)
    cv2.imshow("image", image)
    cv2.waitKey(0)
