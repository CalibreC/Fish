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


class Capture:
    def __init__(
        self, class_name="UnityWndClass", window_name="原神", capture_method="win32api"
    ):
        self.window_name = window_name
        self.class_name = class_name
        self.hwnd = win32gui.FindWindow(self.class_name, self.window_name)
        if self.hwnd is None:
            logger.error("未找到窗口")
            sys.exit(1)

        self.window_rect = None
        self.client_rect = None

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

    def _calc_client_rect(self):
        window_width = self.window_rect[2] - self.window_rect[0]
        window_height = self.window_rect[3] - self.window_rect[1]
        client_width = self.client[2] - self.client[0]
        client_height = self.client[3] - self.client[1]
        align = (window_width - client_width) / 2
        left = int(self.window_rect[0] + align)
        right = int(self.window_rect[2] - align)
        bot = int(self.window_rect[3] - align)
        top_align = window_height - client_height - align
        top = int(self.window_rect[1] + top_align)

        return left, top, right, bot

    def get_window_info(self):
        """

        :return: 窗口句柄，左上角坐标，右下角坐标
        """
        # 如果使用高 DPI 显示器（或 > 100% 缩放尺寸），添加下面一行，否则注释掉
        windll.user32.SetProcessDPIAware()

        # 根据您是想要整个窗口还是只需要 client area 来更改下面的行。
        self.client_rect = win32gui.GetClientRect(self.hwnd)
        logger.info(f"client: {self.client_rect}")
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        logger.info(f"window_rect: {self.window_rect}")

        if self.capture_method == "win32api":
            left, top, right, bot = self.client_rect
        elif self.capture_method == "dxcam":
            left, top, right, bot = self._calc_client_rect()
        else:
            raise ValueError("capture_method should be win32api or dxcam")

        logger.info(f"left: {left}, top: {top}, right: {right}, bot: {bot}")
        return left, top, right, bot

    def _dxcam_capture(self):
        (left, top, right, bot) = self.get_window_info()

        im = self.camera.grab(region=(left, top, right, bot))

        if im is not None:
            # PrintWindow Succeeded
            # im.save("test.png")  # 调试时可打开，不保存图片可节省大量时间（约0.2s）
            return im  # 返回图片
        else:
            logger.error("截图失败")
            return None

    def _win32api_capture(self):
        hwnd, (left, top, right, bot) = Capture.get_window_info()

        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(hwnd)  # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)  # 根据窗口的DC获取mfcDC
        saveDC = mfcDC.CreateCompatibleDC()  # mfcDC创建可兼容的DC

        saveBitMap = win32ui.CreateBitmap()  # 创建bitmap准备保存图片
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)  # 为bitmap开辟空间

        saveDC.SelectObject(saveBitMap)  # 高度saveDC，将截图保存到saveBitmap中

        # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST,
        # 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        # 选择合适的 window number，如0，1，2，3，直到截图从黑色变为正常画面
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
        # win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST,
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
        win32gui.ReleaseDC(hwnd, hwndDC)

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
