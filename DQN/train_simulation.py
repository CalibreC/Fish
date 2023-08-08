#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2023/8/8 0:45
# @Author           : CalibreC
# @Email            : fd98shadow@gmail.com
# @File             : train_simulation.py
# @Description      :
import argparse
import os

import torch
from agent import DQN
from environment import *
from fish_net import FishNet
from render import  *


# from utils.render import *

parser = argparse.ArgumentParser(
    description="Train Genshin finsing simulation with DQN"
)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--n_states", default=3, type=int)
parser.add_argument("--n_actions", default=2, type=int)
parser.add_argument("--step_tick", default=12, type=int)
parser.add_argument("--n_episode", default=400, type=int)
parser.add_argument("--save_dir", default=".\DQN\output", type=str)
parser.add_argument("--resume", default=None, type=str)
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

net = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
if args.resume:
    net.load_state_dict(torch.load(args.resume))

agent = DQN(net, args.batch_size, args.n_states, args.n_actions, memory_capacity=2000)
# env = Fishing_sim(step_tick=args.step_tick, drawer=PltRender())
env = Fishing_simulator(step_tick=args.step_tick, drawer=PltRender())

if __name__ == "__main__":
    # Start training
    print("\nCollecting experience...")
    net.train()
    for i_episode in range(args.n_episode):
        # keyboard.wait('r')
        # play 400 episodes of cartpole game
        state = env.reset()
        ep_r = 0
        while True:
            # 200次训练之后，每20次渲染一次
            if i_episode > 100 and i_episode % 20 == 0:
                env.render()
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
        net.state_dict(), os.path.join(args.save_dir, f"fish_sim_net_{i_episode}.pth")
    )
