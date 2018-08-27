# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
import sys

sys.path.append("D:/RL/DRL-FlappyBird-master/game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80, 1))

# 游戏引擎
def playFlappyBird():
    # Step 1: init BrainDQN
    actions = 2
    brain = BrainDQN(actions)

    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()

    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1, 0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    # observation0 = preprocess(observation0) # 为何初始化brain时使用的图片是[80,80],而不是[80,80,1] => 因为setInitState（）输入的是4张重叠的图片
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while True:
        action = brain.getAction() # 基于当前observation获取下一个action
        nextObservation, reward, terminal = flappyBird.frame_step(action) # 采取行动action
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation, action, reward, terminal) # 学习这个行动带来的回报


def main():
    playFlappyBird()


if __name__ == '__main__':
    main()
