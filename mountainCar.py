import time
start_time = time.time()

import  matplotlib.pyplot as plt

import keras
from keras.layers import Dense
import gym
import numpy as np
from collections import deque
from dqn import DQN
from dqn_scorer import dqnScorerMountainCar

import random

plt.figure()
plt.plot(list(range(10)))


def main(lr=0.001):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    model = keras.Sequential()
    model.add(Dense(128, activation= "relu",input_dim=3, kernel_initializer='normal'))
    model.add(Dense(52, activation= "relu"))
    model.add(Dense(1, kernel_initializer='normal', activation="linear"))
    adam = keras.optimizers.Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=adam)

    gamma = 0.95
    memorySize = 200*100
    dqn = DQN(model, gamma, memorySize,replaysize=64,_env = env)
    dqnScore = dqnScorerMountainCar(dqn,_env=env)
    nrofEpisodes = 1000

    res = np.zeros(shape=(nrofEpisodes, 2))

    for episode in range(nrofEpisodes):
        env.reset()
        action = 0
        obs, _, done, _ = env.step(action)
        #if (episode % 100) == 10:
        if (episode % 100) == 10:
            print("episode " , episode)
            dqnScore.printDistance()
            #dqnScore.plot_cost_to_ßgo_mountain_car()
            #print(res[episode-1,:])
            print("--- %s seconds ---" % (time.time() - start_time))
        iter = 0
        while not done:
            iter += 1
            action = dqn.action(obs)
            new_obs, reward, done, info = env.step(action)
            dqn.add(action,obs,new_obs,reward)
            obs = new_obs
            if(done and (iter<199)):
                print("****Success*****")
            #if(episode % 100) == 10:
            #    env.render()
        dqn.replay()
        env.reset()
        dqnScore.updateResult()
        #res[episode,:] = [np.min(x[:,0]),np.max(x[:,0])]

    dqnScore.plotResults()
    dqnScore.plot_cost_to_go_mountain_car()
    plt.show()


#lr =[0.1,0.01,0.001,0.0001]

main(lr=0.004)



