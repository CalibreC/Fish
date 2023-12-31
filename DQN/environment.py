#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/8 0:46
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : environment.py
# @Description      :
import time
from copy import deepcopy

import cv2
import numpy as np

from DQN.operation import *
from DQN.utils import get_valid_region, match_image
from utils import *


class FishingSimulator(object):
    def __init__(
        self,
        bar_range=(0.18, 0.4),  # 区间长度变化范围
        move_range=(30, 60 * 2),  # 区间移动范围
        resize_freq_range=(15, 60 * 5),
        move_speed_range=(-0.3, 0.3),  # 15个时间计数内，区间移动范围
        tick_count=60,
        step_tick=15,  # 要求连续15个时间计数内，指针在目标区间内
        stop_tick=60 * 15,  # 60*15个时间计数后，one epoch模拟结束
        drag_force=0.4,  # 按下鼠标后，15个时间计数内，指针产生的位移
        down_speed=0.015,  # 15个时间计数内，自动衰减速度
        stable_speed=-0.32,
        drawer=None,
    ):
        self.ticks = None
        self.score = None
        self.move_speed = None
        self.move_tick = None
        self.resize_tick = None
        self.v = None  # drag点击后产生的指针产生的位移
        self.pointer = None
        self.low = None
        self.len = None
        self.bar_range = bar_range
        self.move_range = move_range
        self.resize_freq_range = resize_freq_range
        self.move_speed_range = (
            move_speed_range[0] / tick_count,
            move_speed_range[1] / tick_count,
        )
        self.tick_count = tick_count

        self.step_tick = step_tick
        self.stop_tick = stop_tick
        self.drag_force = drag_force / tick_count
        self.down_speed = down_speed / tick_count
        self.stable_speed = stable_speed / tick_count

        self.drawer = drawer

        self.reset()

    def reset(self):
        self.len = np.random.uniform(*self.bar_range)  # 目标区间长度
        self.low = np.random.uniform(0, 1 - self.len)  # 左边界
        self.pointer = np.random.uniform(0, 1)  # 目标位置
        self.v = 0

        self.resize_tick = 0
        self.move_tick = 0
        self.move_speed = 0

        self.score = 100
        self.ticks = 0

        # 左边界，右边界，目标位置
        return self.low, self.low + self.len, self.pointer

    def drag(self):
        self.v = self.drag_force

    def move_bar(self):
        if self.move_tick <= 0:
            self.move_tick = np.random.uniform(*self.move_range)
            self.move_speed = np.random.uniform(*self.move_speed_range)
        self.low = np.clip(self.low + self.move_speed, a_min=0, a_max=1 - self.len)
        self.move_tick -= 1

    def resize_bar(self):
        if self.resize_tick <= 0:
            self.resize_tick = np.random.uniform(*self.resize_freq_range)
            self.len = min(np.random.uniform(*self.bar_range), 1 - self.low)
        self.resize_tick -= 1

    def tick(self):
        self.ticks += 1
        if self.low < self.pointer < self.low + self.len:  # 目标在区间内
            self.score += 1
        else:
            self.score -= 1

        # 超出本次时间上限|分数过低，one epoch结束
        if self.ticks > self.stop_tick or self.score <= -100000:
            return True

        self.pointer += self.v
        # 将指针限制在[0,1]区间内
        self.pointer = np.clip(self.pointer, a_min=0, a_max=1)
        self.v = max(self.v - self.down_speed, self.stable_speed)

        self.move_bar()
        self.resize_bar()
        return False

    def do_action(self, action):
        if action == 1:
            self.drag()

    def get_state(self):
        return self.low, self.low + self.len, self.pointer

    def step(self, action):
        self.do_action(action)

        done = False
        score_before = self.score
        for x in range(self.step_tick):  # 15时间计数
            if self.tick():  # 执行action
                done = True
        return self.get_state(), (self.score - score_before) / self.step_tick, done

    def render(self):
        if self.drawer:
            self.drawer.draw(self.low, self.low + self.len, self.pointer, self.ticks)


class Fishing:
    def __init__(
        self, delay=0.1, max_step=100, capture_method=None, show_detection=False
    ):
        self.last_score = 0
        self.reward = 0
        self.zero_count = 0
        self.fish_start = False  # 是否开始钓鱼
        self.step_count = 0
        self.image = None  # 包含钓鱼进度条

        self.show_detection = show_detection

        # reading images
        self.full_image = cv2.imread("./imgs/test/test.jpg")
        self.t_l = cv2.imread("./imgs/target_left.png")
        self.t_r = cv2.imread("./imgs/target_right.png")
        self.t_n = cv2.imread("./imgs/target_now.png")
        self.bar_template = cv2.imread("./imgs/bar_template.png")
        self.bite = cv2.imread("./imgs/bite.png", cv2.IMREAD_GRAYSCALE)
        self.fishing = cv2.imread("./imgs/fishing.png", cv2.IMREAD_GRAYSCALE)
        self.exit = cv2.imread("./imgs/exit.png")

        self.delay = delay
        self.max_step = max_step
        self.count = 0

        # state
        self.add_vec = [0, 2, 0, 2, 0, 2]

        # score
        self.r_ring = 21
        self.std_color = np.array([192, 255, 255])

        # camera
        self.camera = capture_method

        # mouse, keyboard
        self.mouse = MouseOperation(self.camera.window)

    def reset(self):
        # TODO: 重置钓鱼进度条
        self.image = get_valid_region(self.full_image, None)

        return self._get_status()

    def _list_add(self, li, num):
        return [x + y for x, y in zip(li, num)]

    def _scale(self, x):
        # 484是整个进度条的长度，可以参考imgs里 16.jpg 500.jpg
        return (x - 5 - 10) / 484  # 5是左边界，10是左边界到第一个刻度的距离

    def _get_status(self):
        # TODO: 从钓鱼进度条中获取状态
        # bar_image.shape 32, 516, 3
        bar_image = self.image[2:34, :, :]

        bbox_l = match_image(bar_image, self.t_l)
        bbox_r = match_image(bar_image, self.t_r)
        bbox_n = match_image(bar_image, self.t_n)

        bbox_l = tuple(self._list_add(bbox_l, self.add_vec))
        bbox_r = tuple(self._list_add(bbox_r, self.add_vec))
        bbox_n = tuple(self._list_add(bbox_n, self.add_vec))

        if self.show_detection:
            img = deepcopy(bar_image)
            cv2.rectangle(img, bbox_l[:2], bbox_l[2:4], (255, 0, 0), 1)  # 画出矩形位置
            cv2.rectangle(img, bbox_r[:2], bbox_r[2:4], (0, 255, 0), 1)  # 画出矩形位置
            cv2.rectangle(img, bbox_n[:2], bbox_n[2:4], (0, 0, 255), 1)  # 画出矩形位置
            font_face = cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = 1
            thickness = 1
            cv2.putText(
                img,
                str(self.last_score),
                (257 + 30, 72),
                fontScale=font_scale,
                fontFace=font_face,
                thickness=thickness,
                color=(0, 255, 255),
            )
            cv2.putText(
                img,
                str(self.reward),
                (257 + 30, 87),
                fontScale=font_scale,
                fontFace=font_face,
                thickness=thickness,
                color=(255, 255, 0),
            )
            cv2.imwrite(f"./img_tmp/{self.count}.jpg", img)

        self.count += 1

        # 归一化
        return self._scale(bbox_l[4]), self._scale(bbox_r[4]), self._scale(bbox_n[4])

    def render(self):
        cv2.imshow("image", self.image)
        cv2.waitKey(1)
        pass

    def step(self, action):
        self._do_action(action)

        time.sleep(self.delay)

        # 默认1080p
        image = self.camera.capture()
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        self.image = get_valid_region(image, None)
        self.step_count += 1

        score = self._get_score()
        if score > 0:
            self.fish_start = True
            self.zero_count = 0
        else:
            self.zero_count += 1

        self.reward = score - self.last_score
        self.last_score = score

        return (
            self._get_status(),
            self.reward,
            (
                self.step_count > self.max_step
                or (self.zero_count >= 15 and self.fish_start)
                or score > 176
            ),
        )

    def _do_action(self, action):
        if action == 1:
            # self._drag()
            pass

    def _get_score(self):
        """
        由圆形框确认分数
        """
        cx, cy = 247 + 10, 72  # 圆心
        for x in range(4, 360, 2):  # 4-360度，每隔2度
            px = int(cx + self.r_ring * np.sin(np.deg2rad(x)))
            py = int(cy - self.r_ring * np.cos(np.deg2rad(x)))
            if np.mean(np.abs(self.image[py, px, :] - self.std_color)) > 5:
                return x // 2 - 2
        return 360 // 2 - 2

    def _drag(self):
        self.mouse.rel_click(1630, 995)
