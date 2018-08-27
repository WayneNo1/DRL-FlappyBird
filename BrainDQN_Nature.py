# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100

try:
    tf.mul
except:
    # For new version of tensorflow
    # tf.mul has been removed in new version of tensorflow
    # Using tf.multiply to replace tf.mul
    tf.mul = tf.multiply


class BrainDQN:
    # 基本架构 1.初始化网络
    def __init__(self, actions):
        # init replay memory
        self.replayMemory = deque()

        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions

        # init Q network
        self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2 = self.createQNetwork()

        # init Target Q Network
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    # 基本架构 2.创建Q-Net
    # CNN网络全连接层的输出即为每个action的Q值
    def createQNetwork(self):
        # 1. network weights
        # 卷积层1：[80,80,4] ->（stride:4） [20,20,32] ->(mal_pooling:2*2) [10,10,32]
        W_conv1 = self.weight_variable([8, 8, 4, 32]) # 卷积层1的卷积核结构,in 4个channel，out 32个channel
        b_conv1 = self.bias_variable([32])

        # 卷积层2：[10,10,32] ->（stride:2） [5,5,64]
        W_conv2 = self.weight_variable([4, 4, 32, 64]) # 卷积层2的卷积核结构
        b_conv2 = self.bias_variable([64])

        # 卷积层3：[5,5,64] ->（stride:1） [5,5,64]
        W_conv3 = self.weight_variable([3, 3, 64, 64]) # 卷积层3的卷积核结构
        b_conv3 = self.bias_variable([64])

        # 全连接层1：
        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])

        # 全连接层2：
        W_fc2 = self.weight_variable([512, self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # 2. input layer
        stateInput = tf.placeholder("float", [None, 80, 80, 4])

        # 3. hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput, W_conv1, 4) + b_conv1) # [80,80,4] ->（stride:4） [20,20,32]
        h_pool1 = self.max_pool_2x2(h_conv1) # [20,20,32] ->(mal_pooling:2*2) [10,10,32]

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2) # [10,10,32] ->（stride:2） [5,5,64]

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3) # [5,5,64] -> [5,5,64]

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600]) # [None, 1600]
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1) # [None, 512]

        # Q Value layer
        QValue = tf.matmul(h_fc1, W_fc2) + b_fc2 # [None, 2]

        return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2

    # 更新Target_net 网络参数
    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    # 构建训练过程计算逻辑
    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float", [None])

        Q_Action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices=1) # Q估计（使用eval_net，根据state s）
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    # 基本架构 3.训练Q网络
    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        # 一条记忆[currentState, action, reward, newState, terminal]
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT: nextState_batch}) # 使用target_net，根据nextState s_，用于计算Q现实
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step=self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    # 基本架构 4.一条记忆
    def setPerception(self, nextObservation, action, reward, terminal):
        # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = np.append(self.currentState[:, :, 1:], nextObservation, axis=2) # 一个state有四张图片，对应CNN图片中的4个channel，新状态为前3张加上最新一张
        self.replayMemory.append((self.currentState, action, reward, newState, terminal)) # 记忆库中添加一条记忆
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE: # 记忆库积累一定记忆之后开始训练
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe" # observe阶段积累记忆（100个）
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state, \
              "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    # 基本架构 5.根据当前状态，选择action
    def getAction(self):
        QValue = self.QValue.eval(feed_dict={self.stateInput: [self.currentState]})[0] # Q估计（输入只有一个样本，输出也是只有一个Q值向量）
        action = np.zeros(self.actions)
        action_index = 0
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actions)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            # input_actions[0] == 1: do nothing
            # input_actions[1] == 1: flap the bird
            action[0] = 1  # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    # 基本架构 6.设置初始状态
    def setInitState(self, observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis=2)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
