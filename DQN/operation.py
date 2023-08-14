import time

import win32api
import win32con
from window import Window


class MouseOperation:
    def __init__(self, window=None):
        self.MOUSE_LEFT = 0
        self.MOUSE_MID = 1
        self.MOUSE_RIGHT = 2

        self.mouse_list_down = [
            win32con.MOUSEEVENTF_LEFTDOWN,
            win32con.MOUSEEVENTF_MIDDLEDOWN,
            win32con.MOUSEEVENTF_RIGHTDOWN,
        ]
        self.mouse_list_up = [
            win32con.MOUSEEVENTF_LEFTUP,
            win32con.MOUSEEVENTF_MIDDLEUP,
            win32con.MOUSEEVENTF_RIGHTUP,
        ]

        self.window = window

    def click(self, x, y):
        """
        鼠标点击
        :param x: x坐标
        :param y: y坐标
        :param delay: 延迟时间
        :return:
        """
        win32api.SetCursorPos((x, y))
        self._down(x, y)
        self._up(x, y)

    def _up(self, x, y):
        win32api.mouse_event(self.mouse_list_up[self.MOUSE_LEFT], x, y, 0, 0)

    def _down(self, x, y):
        win32api.mouse_event(self.mouse_list_down[self.MOUSE_LEFT], x, y, 0, 0)

    def rel_click(self, x, y):
        """
        以1080p截图左上角为坐标系
        :param x: x坐标
        :param y: y坐标
        :param delay: 延迟时间
        :return:
        """
        real_left, real_top = self._calc_real_pos(x, y)
        self.click(real_left, real_top)

    def _calc_real_pos(self, x, y):
        left, top, right, bot = self.window.get_window_info("dxcam")
        real_x = left + x
        real_y = top + y
        return real_x, real_y


class KeyboardOperation:
    def __init__(self):
        pass


if __name__ == '__main__':
    # Genshin = Window(class_name="UnityWndClass", window_name="原神")
    window = Window(class_name=None, window_name="向日葵远程控制")
    mouse = MouseOperation(window)
    # mouse.click(1280, 720)
    mouse.rel_click(50, 320)
