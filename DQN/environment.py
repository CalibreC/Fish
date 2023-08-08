#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/8 0:46
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : environment.py
# @Description      :
import numpy as np


class Fishing_simulator(object):
    def __init__(
        self,
        bar_range=(0.18, 0.4),                          # 区间长度变化范围
        move_range=(30, 60 * 2),                        # 区间移动范围
        resize_freq_range=(15, 60 * 5),
        move_speed_range=(-0.3, 0.3),                   # 15个时间计数内，区间移动范围
        tick_count=60,
        step_tick=15,                                   # 要求连续15个时间计数内，指针在目标区间内
        stop_tick=60 * 15,                              # 60*15个时间计数后，one epoch模拟结束
        drag_force=0.4,                                 # 按下鼠标后，15个时间计数内，指针产生的位移
        down_speed=0.015,                               # 15个时间计数内，自动衰减速度
        stable_speed=-0.32,
        drawer=None,
    ):
        self.ticks = None
        self.score = None
        self.move_speed = None
        self.move_tick = None
        self.resize_tick = None
        self.v = None                                   # drag点击后产生的指针产生的位移
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
        if self.low < self.pointer < self.low + self.len:   # 目标在区间内
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
        for x in range(self.step_tick): # 15时间计数
            if self.tick():             # 执行action
                done = True
        return self.get_state(), (self.score - score_before) / self.step_tick, done

    def render(self):
        if self.drawer:
            self.drawer.draw(self.low, self.low + self.len, self.pointer, self.ticks)
