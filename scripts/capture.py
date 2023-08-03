#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/3 1:43
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : capture.py
# @Description      :
import sys
import time
from ctypes import windll

import cv2
import dxcam
import keyboard
import numpy as np
import win32api
import win32con
import win32gui
import win32ui
from loguru import logger
from PIL import Image

# window_name = "原神"
window_name = "Notepad"
capture_method = "win32api"
# capture_method = "dxcam"

# def capture_image():
#     camera = dxcam.create(output_color="BGRA")
#     camera.start(video_mode=False, region=region)
#     image = camera.get_latest_frame()
#     camera.stop()
#     cv2.imwrite("image.png", image)

# 截取窗口的图像
# def capture_video():
#     target_fps = 30
#     camera = dxcam.create(output_color="BGRA")
#     camera.start(target_fps=target_fps, video_mode=True, region=region)
#     writer = cv2.VideoWriter(
#         "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (w, h)
#     )
#     for i in range(60):
#         image = camera.get_latest_frame()
#         if image is not None:
#             print(image.shape)
#             writer.write(image)
#         else:
#             print("no image")
#     camera.stop()
#     writer.release()


def get_window_info():
    """

    :return: 窗口句柄，左上角坐标，右下角坐标
    """
    hwnd = win32gui.FindWindow(window_name, None)

    # 如果使用高 DPI 显示器（或 > 100% 缩放尺寸），添加下面一行，否则注释掉
    windll.user32.SetProcessDPIAware()

    # 根据您是想要整个窗口还是只需要 client area 来更改下面的行。
    client = win32gui.GetClientRect(hwnd)
    logger.info(f"client: {client}")
    window = win32gui.GetWindowRect(hwnd)
    logger.info(f"window: {window}")

    left = client[0]
    top = client[1]
    right = client[2]
    bot = client[3]

    # left, top, right, bot = client[0], client[1], client[2], client[3]

    if capture_method == "win32api":
        pass
    elif capture_method == "dxcam":
        window_width = window[2] - window[0]
        window_height = window[3] - window[1]
        client_width = client[2] - client[0]
        client_height = client[3] - client[1]
        align = (window_width - client_width) / 2
        left = int(window[0] + align)
        right = int(window[2] - align)
        bot = int(window[3] - align)
        top_align = window_height - client_height - align
        top = int(window[1] + top_align)
    else:
        raise ValueError("capture_method should be win32api or dxcam")

    logger.info(f"left: {left}, top: {top}, right: {right}, bot: {bot}")
    return hwnd, (left, top, right, bot)


def dxcam_init():
    camera = dxcam.create(
        device_idx=0, output_color="BGRA", output_idx=0
    )  # returns a DXCamera instance on primary monitor
    return camera


camera = dxcam_init()


def dxcam_capture():
    hwnd, (left, top, right, bot) = get_window_info()

    im = camera.grab(region=(left, top, right, bot))

    if im is not None:
        # PrintWindow Succeeded
        # im.save("test.png")  # 调试时可打开，不保存图片可节省大量时间（约0.2s）
        return im  # 返回图片
    else:
        logger.error("截图失败")
        return None


def win32api_capture():
    hwnd, (left, top, right, bot) = get_window_info()

    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)  # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)  # 根据窗口的DC获取mfcDC
    saveDC = mfcDC.CreateCompatibleDC()  # mfcDC创建可兼容的DC

    saveBitMap = win32ui.CreateBitmap()  # 创建bitmap准备保存图片
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)  # 为bitmap开辟空间

    saveDC.SelectObject(saveBitMap)  # 高度saveDC，将截图保存到saveBitmap中

    # 选择合适的 window number，如0，1，2，3，直到截图从黑色变为正常画面
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        "RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]), bmpstr, "raw", "BGRX", 0, 1
    )

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        # PrintWindow Succeeded
        # im.save("test.png")  # 调试时可打开，不保存图片可节省大量时间（约0.2s）
        return im  # 返回图片
    else:
        logger.error("截图失败")
        return None


def video_capture():
    if capture_method == "win32api":
        capture = win32api_capture
    elif capture_method == "dxcam":
        capture = dxcam_capture
    else:
        raise ValueError("capture_method should be win32api or dxcam")

    while True:
        start = time.time()
        img = capture()
        if img is not None:
            image = np.asarray(img)
            # img = torch.from_numpy(img).cuda()
            # 推理，后处理
            cv2.imshow("image", image)
            cv2.waitKey(1)
        else:
            time.sleep(0.04)
        end = time.time()
        fps = 1 / np.round(end - start, 3)
        print(f"Frames Per Second : {fps}")


if __name__ == "__main__":
    logger.remove()  # 删除自动产生的handler
    handle_id = logger.add(sys.stderr, level="WARNING")  # 添加一个可以修改控制的handler
    # 模型初始化

    video_capture()
