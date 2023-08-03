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
import keyboard
from ctypes import windll

window_name = "原神"


# 找到窗口，并将窗口置顶
# hwnd = win32gui.GetDesktopWindow()
hwnd = win32gui.FindWindow(None, window_name)



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
time.sleep(0.1)

scale = 1.1

x = rect[0]
y = rect[1]
w = rect[2] - x + 1
h = rect[3] - y + 1
# right = int(right*scale)
# bottom = int(bottom*scale)
# weight = int((right - left) * scale)
# height = int((bottom - top) * scale)

# region = [left, top, weight, height]
# w = int(w * scale)
# h = int(h * scale)
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

    print("before")
    keyboard.wait('r')
    print("OK")
    win32gui.SetForegroundWindow(hwnd)
    # ... Do Something
    while True:
        start = time.time()
        img = camera.get_latest_frame()
        # image = torch.from_numpy(img).cuda()
        print(img.shape)
        cv2.imshow("image", img)
        cv2.waitKey(1)
        end = time.time()
        fps = 1 / np.round(end - start, 3)
        # print(f"Frames Per Second : {fps}")

    camera.stop()


def cap_raw(region=None, fmt='RGB'):
    print("before")
    keyboard.wait('r')
    print("OK")
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(1)
    # hwnd = win32gui.GetDesktopWindow()
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()

    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)

    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (x, y), win32con.SRCCOPY)
    # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype="uint8")
    img.shape = (h, w, 4)

    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    if fmt == 'BGR':
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)
    if fmt == 'RGB':
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError('Cannot indetify this fmt')


#局部截图
def window_capturex():
    # proportion = round(win32print.GetDeviceCaps(win32gui.GetDC(0), win32con.DESKTOPHORZRES)/win32api.GetSystemMetrics(0), 2)
    # print(proportion)
    print("before")
    keyboard.wait('r')
    print("OK")
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(0.05)
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    # left=int(left*proportion)
    # top=int(top*proportion)
    # right=int(right*proportion)
    # bot=int(bot*proportion)
    w = right - left
    h = bot - top
    hWndDC = win32gui.GetWindowDC(win32gui.GetDesktopWindow())
    mfcDC = win32ui.CreateDCFromHandle(hWndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC,w,h)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0,0), (w,h), mfcDC, (left, top), win32con.SRCCOPY)
    try:
        saveBitMap.SaveBitmapFile(saveDC, "tempcap.bmp")
    except Exception as e:
        print("错误")
        pass
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(win32gui.GetDesktopWindow(), hWndDC)

def photo_capture():

    hwnd = win32gui.FindWindow(None, window_name)  # 获取窗口的句柄

    # 如果使用高 DPI 显示器（或 > 100% 缩放尺寸），添加下面一行，否则注释掉
    windll.user32.SetProcessDPIAware()

    # Change the line below depending on whether you want the whole window
    # or just the client area.
    # 根据您是想要整个窗口还是只需要 client area 来更改下面的行。
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    # left, top, right, bot = win32gui.GetWindowRect(hwnd)
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
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        # PrintWindow Succeeded
        # im.save("test.png")  # 调试时可打开，不保存图片可节省大量时间（约0.2s）
        cv2.imshow("image", np.array(im))
        cv2.waitKey(0)
        return im  # 返回图片
    else:
        print("fail")

if __name__ == "__main__":
    # 同步播放并推理
    # capture_image()
    # capture_video()
    # grab_screen_dxcam()
    # cv2.imshow("image", cap_raw())
    # cv2.waitKey(0)
    # window_capturex()
    photo_capture()