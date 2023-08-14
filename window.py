import sys
import time
from ctypes import windll

import cv2
import numpy as np
import win32gui
from loguru import logger


class Window:
    def __init__(self, class_name="UnityWndClass", window_name="原神"):
        self.class_name = class_name
        self.window_name = window_name
        self.hwnd = win32gui.FindWindow(self.class_name, self.window_name)
        if self.hwnd is None:
            logger.error("未找到窗口")
            sys.exit(1)

        self.window_rect = None
        self.client_rect = None

    def _calc_client_rect(self):
        window_width = self.window_rect[2] - self.window_rect[0]
        window_height = self.window_rect[3] - self.window_rect[1]
        client_width = self.client_rect[2] - self.client_rect[0]
        client_height = self.client_rect[3] - self.client_rect[1]
        align = (window_width - client_width) / 2
        left = int(self.window_rect[0] + align)
        right = int(self.window_rect[2] - align)
        bot = int(self.window_rect[3] - align)
        top_align = window_height - client_height - align
        top = int(self.window_rect[1] + top_align)

        return left, top, right, bot

    def get_window_info(self, capture_method):
        """

        :return: 左上角坐标，右下角坐标
        """
        # 如果使用高 DPI 显示器（或 > 100% 缩放尺寸），添加下面一行，否则注释掉
        windll.user32.SetProcessDPIAware()

        # 根据您是想要整个窗口还是只需要 client area 来更改下面的行。
        self.client_rect = win32gui.GetClientRect(self.hwnd)
        logger.info(f"client_rect: {self.client_rect}")
        self.window_rect = win32gui.GetWindowRect(self.hwnd)
        logger.info(f"window_rect: {self.window_rect}")

        if capture_method == "win32api":
            left, top, right, bot = self.client_rect
        elif capture_method == "dxcam":
            left, top, right, bot = self._calc_client_rect()
            logger.info("left, top, right, bot: ", left, top, right, bot)
        else:
            raise ValueError("capture_method should be win32api or dxcam")

        return left, top, right, bot
