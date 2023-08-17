#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/13 0:59
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : train_genshin.py
# @Description      :
import argparse
import os
import sys
import winsound

import keyboard
import torch
from loguru import logger

from capture import Capture
from DQN.agent import DQN
from DQN.environment import Fishing
from DQN.fish_net import FishNet
from window import Window


def logger_setting():
    logger.remove()  # 删除自动产生的handler
    handle_id = logger.add(sys.stderr, level="WARNING")  # 添加一个可以修改控制的handler


def make_parser():
    parser = argparse.ArgumentParser(description="Train Genshin fishing with DQN")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_states", default=3, type=int)
    parser.add_argument("--n_actions", default=2, type=int)
    parser.add_argument("--step_tick", default=12, type=int)
    parser.add_argument("--n_episode", default=400, type=int)
    parser.add_argument("--save_dir", default="./output", type=str)
    parser.add_argument(
        "--resume", default="./DQN/output/fish_sim_net_399.pth", type=str
    )
    return parser


if __name__ == "__main__":
    logger_setting()
    args = make_parser().parse_args()

    Genshin = Window(class_name="UnityWndClass", window_name="原神")
    DirectX = Capture(window=Genshin, capture_method="dxcam")

    net = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    if args.resume:
        net.load_state_dict(torch.load(args.resume))

    agent = DQN(
        net, args.batch_size, args.n_states, args.n_actions, memory_capacity=1000
    )
    env = Fishing(delay=0.1, max_step=150, capture_method=DirectX)

    # Start training
    print("\nCollecting experience...")
    net.train()

    print("Press 'r' to start training.")
    winsound.Beep(500, 500)
    keyboard.wait("r")

    for i_episode in range(args.n_episode):
        state = env.reset()
        ep_r = 0
        while True:
            # if i_episode > 200 and i_episode % 20 == 0:
            env.render()
            print("state: ", state)
            # take action based on the current state
            action = agent.choose_action(state)
            # obtain the reward and next state and some other information
            state_, reward, done = env.step(action)

            # store the transitions of states
            agent.store_transition(state, action, reward, state_, int(done))

            ep_r += reward
            # if the experience repaly buffer is filled, DQN begins to learn or update
            # its parameters.
            if agent.memory_counter > agent.memory_capacity:
                agent.train_step()
                if done:
                    print("Ep: ", i_episode, " |", "Ep_r: ", round(ep_r, 2))

            if done:
                # if game is over, then skip the while loop.
                break
            # use next state to update the current state.
            state = state_
        torch.save(
            net.state_dict(),
            os.path.join(args.save_dir, f"fish_ys_net_{i_episode}.pth"),
        )
