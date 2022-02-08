# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:13:35 2021

@author: Tiago

"""

import random
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

env = gym.make("BreakoutNoFrameskip-v4")
height, width, channels = env.observation_space.shape
actions = env.action_space.n
print("The action size is : ", env.unwrapped.get_action_meanings())
env._max_episode_steps = 1000
episodes = 1

# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done: 
#         env.render()
#         action = random.choice([0,1,2,3])
#         n_state, reward, done, info = env.step(action) 
#         score += reward
#     print('Episode: {} Score:{}'.format(episode, score))
# env.close()


def build_model(height, width,channels, actions):
    model = Sequential()
    model.add(Convolution2D(32,(8,8), strides=(4,4), activation ='relu', input_shape=(3,height,width,channels)))
    model.add(Convolution2D(64,(4,4), strides=(4,4), activation ='relu'))
    model.add(Convolution2D(32,(3,3), activation ='relu'))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
 

model = build_model(height, width, channels, actions)
print(model.summary())
def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps',value_max=1.,value_min=.1,value_test=.2, nb_steps=10)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn =DQNAgent(model=model, memory=memory, policy=policy,enable_dueling_network=True,
                  dueling_type='avg',nb_actions=actions, nb_steps_warmup=1000)
    return dqn  
dqn = build_agent(model,actions)
dqn.compile(Adam(lr=1e-4))
#dqn.fit(env,nb_steps=50000, visualize=False, verbose=2 )

#dqn.save_weights('saved_weights/dqn_weights.h5f')
dqn.load_weights('saved_weights/dqn_weights.h5f')
scores = dqn.test(env,nb_episodes=1000,visualize=True)
print(np.mean(scores.history['episode_reward']))





