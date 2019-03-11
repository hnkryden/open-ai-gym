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


def main(lr=0.001, episodeMemory = 100, replaySize=64,gamma=0.95):
    np.random.seed(0)
    env = gym.make('MountainCar-v0')
    model = keras.Sequential()
    model.add(Dense(128, activation= "relu",input_dim=3, kernel_initializer='normal'))
    model.add(Dense(52, activation= "relu"))
    model.add(Dense(1, kernel_initializer='normal', activation="linear"))
    adam = keras.optimizers.Adam(lr=lr)
    model.compile(loss='mean_squared_error', optimizer=adam)

    #gamma = 0.95
    memorySize = 200*episodeMemory
    dqn = DQN(model, gamma, memorySize,replaysize=replaySize,_env = env)
    dqnScore = dqnScorerMountainCar(dqn,_env=env)
    nrofEpisodes = 1001
    #nrofEpisodes = 20

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
            if(done and (iter<199)):
                reward = (200 - iter)/10
                print("****Success*****", -iter)

            dqn.add(action,obs,new_obs,reward)
            obs = new_obs

            #if(episode % 100) == 10:
            #    env.render()j
        dqn.replay()
        env.reset()
        dqnScore.updateResult(iter)
        #res[episode,:] = [np.min(x[:,0]),np.max(x[:,0])]
    title = "eps_%d_mem_%d_rep_%d_gamma_%d" % (nrofEpisodes,episodeMemory,replaySize,gamma*100)
    dqnScore.plotResults(title)
    dqnScore.plot_cost_to_go_mountain_car(title)
    #plt.show()

#lr =[0.1,0.01,0.001,0.0001]

#main(lr=0.005,episodeMemory=5,replaySize=64,gamma=0.95)
#main(lr=0.004,episodeMemory=500,replaySize=64,gamma=0.95)

main(lr=0.004,episodeMemory=100,replaySize=128,gamma=0.95)
#main(lr=0.004,episodeMemory=100,replaySize=128,gamma=0.95)

#main(lr=0.004,episodeMemory=100,replaySize=64,gamma=0.9)
#main(lr=0.004,episodeMemory=100,replaySize=64,gamma=0.99)

