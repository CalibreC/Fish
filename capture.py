import sys
import time
from ctypes import windll

import cv2
import dxcam
import numpy as np
import win32gui
import win32ui
from loguru import logger
from PIL.Image import Image
from window import Window

class Capture:
    def __init__(
            self, window, capture_method="win32api"
    ):
        self.window = window

        self.capture_method = capture_method
        if self.capture_method == "dxcam":
            self.camera = dxcam.create(device_idx=0, output_color="BGRA", output_idx=0)

            if self.camera is None:
                logger.error("未找到数据源")
                sys.exit(1)

            self.capture = self._dxcam_capture
        elif self.capture_method == "win32api":
            self.capture = self._win32api_capture
        else:
            logger.error("未知的截图方式")
            sys.exit(1)

    def _dxcam_capture(self):
        (left, top, right, bot) = self.window.get_window_info(self.capture_method)

        im = self.camera.grab(region=(left, top, right, bot))

        if im is not None:
            # PrintWindow Succeeded
            # im.save("test.png")  # 调试时可打开，不保存图片可节省大量时间（约0.2s）
            return im  # 返回图片
        else:
            logger.error("截图失败")
            return None

    def _win32api_capture(self):
        left, top, right, bot = self.window.get_window_info(self.capture_method)

        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(self.window.hwnd)  # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)  # 根据窗口的DC获取mfcDC
        saveDC = mfcDC.CreateCompatibleDC()  # mfcDC创建可兼容的DC

        saveBitMap = win32ui.CreateBitmap()  # 创建bitmap准备保存图片
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)  # 为bitmap开辟空间

        saveDC.SelectObject(saveBitMap)  # 高度saveDC，将截图保存到saveBitmap中

        # win32gui.SetWindowPos(self.window.hwnd, win32con.HWND_TOPMOST,
        # 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        # 选择合适的 window number，如0，1，2，3，直到截图从黑色变为正常画面
        result = windll.user32.PrintWindow(self.window.hwnd, saveDC.GetSafeHdc(), 2)
        # win32gui.SetWindowPos(self.window.hwnd, win32con.HWND_NOTOPMOST,
        # 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.window.hwnd, hwndDC)

        # if result == 1:
        # PrintWindow Succeeded
        # im.save("test.png")  # 调试时可打开，不保存图片可节省大量时间（约0.2s）
        return im  # 返回图片
        # else:
        #     logger.error("截图失败")
        #     return None

    def video_capture(self, window_name, capture_method):
        while True:
            start = time.time()
            img = self.capture()
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
