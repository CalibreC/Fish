#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/3 1:43
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : capture.py
# @Description      :

import time

import cv2
import dxcam
import numpy as np
import win32api
import win32con
import win32gui
import win32ui
from PIL import Image

window_name = "原神"


# 找到窗口，并将窗口置顶
# hwnd = win32gui.GetDesktopWindow()
hwnd = win32gui.FindWindow(None, window_name)


win32gui.SetForegroundWindow(hwnd)
# win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
# win32gui.SetWindowPos(
#           hwnd,
#           win32.HWND_NOTOPMOST,
#           0, 0, 0, 0,
#           win32con.SWP_SHOWWINDOW |
#           win32con.SWP_NOSIZE |
#           win32con.SWP_NOMOVE)

rect = win32gui.GetWindowRect(hwnd)
print(rect)
time.sleep(1)

scale = 1.5

x = rect[0]
y = rect[1]
w = rect[2] - x
h = rect[3] - y
# right = int(right*scale)
# bottom = int(bottom*scale)
# weight = int((right - left) * scale)
# height = int((bottom - top) * scale)

# region = [left, top, weight, height]
region = (x, y, w, h)
print(region)


# 截取窗口的图像
def capture_video():
    target_fps = 30
    camera = dxcam.create(output_color="BGRA")
    camera.start(target_fps=target_fps, video_mode=True, region=region)
    writer = cv2.VideoWriter(
        "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (w, h)
    )
    for i in range(60):
        image = camera.get_latest_frame()
        if image is not None:
            print(image.shape)
            writer.write(image)
        else:
            print("no image")
    camera.stop()
    writer.release()


def capture_image():
    camera = dxcam.create(output_color="BGRA")
    camera.start(video_mode=False, region=region)
    image = camera.get_latest_frame()
    camera.stop()
    cv2.imwrite("image.png", image)


def grab_screen_dxcam():
    camera = dxcam.create(
        device_idx=0, output_color="BGRA"
    )  # returns a DXCamera instance on primary monitor
    camera.start(
        region=(x, y, w, h), target_fps=30, video_mode=True
    )  # Optional argument to capture a region

    # ... Do Something
    while True:
        start = time.time()
        img = camera.get_latest_frame()
        # image = torch.from_numpy(img).cuda()
        cv2.imshow("image", img)
        cv2.waitKey(1)
        end = time.time()
        fps = 1 / np.round(end - start, 3)
        print(f"Frames Per Second : {fps}")

    camera.stop()


if __name__ == "__main__":
    # 同步播放并推理
    # capture_image()
    # capture_video()
    grab_screen_dxcam()
