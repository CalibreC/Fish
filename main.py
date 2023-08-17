#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/3 1:43
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : capture.py
# @Description      :
import argparse
import sys
import time

import cv2
import keyboard
import numpy as np
from loguru import logger

from capture import Capture
from load_model import load_model
from postprocess import non_max_suppression, postprocess
from preprocess import preprocess
from window import Window


def logger_setting():
    logger.remove()  # 删除自动产生的handler
    handle_id = logger.add(sys.stderr, level="WARNING")  # 添加一个可以修改控制的handler


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logger-level",
        default="info",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    parser.add_argument("--name", default="原神", type=str, help="选择游戏名，默认为原神")
    parser.add_argument(
        "--method",
        default="dxcam",
        type=str,
        choices=["win32api", "dxcam"],
        help="选择截图方式，默认为dxcam",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        choices=["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="选择需要用于推理的设备，默认为cuda:0",
    )
    parser.add_argument(
        "--data",
        default="data/fish.yaml",
        type=str,
        help="选择需要用于推理的标签，默认为data/fish.yaml",
    )
    return parser


if __name__ == "__main__":
    logger_setting()
    args = make_parser().parse_args()

    Genshin = Window(class_name="UnityWndClass", window_name=args.name)
    Camera = Capture(window=Genshin, capture_method="dxcam")

    model, class_names = load_model()

    logger.info("Ready to detect. Press 'r' to start")
    keyboard.wait("r")

    while True:
        start = time.time()
        img = Camera.capture()
        if img is not None:
            torch_image, cv_image = preprocess(img)

            # inference
            pred = model(torch_image, augment=False, visualize=False)[0]

            # NMS
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

            # postprocess
            cv_image = postprocess(pred, class_names, torch_image, cv_image)

            cv2.imshow("image", cv_image)
            cv2.waitKey(1)
        else:
            time.sleep(0.04)
        end = time.time()
        fps = 1 / np.round(end - start, 3)
        print(f"Frames Per Second : {fps}")
