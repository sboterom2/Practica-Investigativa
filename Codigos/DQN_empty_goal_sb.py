from gfootball.env.config import Config
import gfootball.env as football_env
import gym
from stable_baselines3.common.env_checker import check_env
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import gfootball 
import math
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from gfootball.env import create_environment

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)
        # Otras inicializaciones si es necesari
    def step(self, action):
        # Realiza la acción en el entorno original
        next_state, reward, done, info = self.env.step(action)
        # Calcula la recompensa adicional basada en el estado
        extra_reward = self.calculate_extra_reward(next_state,action)    
        # Suma la recompensa extra a la recompensa original
        reward += extra_reward        
        return next_state, reward, done, info

    def calculate_extra_reward(self, state, action):
        custom_reward = 0
        if action == 5 and state[2] < 0.65 and (-0.2 < state[3] < 0.2):
            custom_reward = custom_reward + 0.2  # Ejemplo de recompensa por avanzar hacia la portería
        if state[88]<0:
            custom_reward=custom_reward - 0.5
        if action in [1, 2, 8] and state[2] < 0.65 :
            custom_reward = custom_reward - 0.2    # Ejemplo de penalización por alejarse de la portería
        
        return custom_reward

env = football_env.create_environment(env_name ='academy_empty_goal',render=True,representation='simple115v2')
env=RewardWrapper(env)

model = DQN("MlpPolicy", env, learning_rate=0.001, buffer_size=10000, learning_starts=1000, batch_size=128, gamma=0.99)

# Entrenar el modelo
model.learn(total_timesteps=10000)