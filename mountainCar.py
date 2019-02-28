
import  matplotlib.pyplot as plt

import keras
from keras.layers import Dense
import gym
import numpy as np
env = gym.make('MountainCar-v0')

env.reset()
model = keras.Sequential()
model.add(Dense(24, activation= "relu",input_dim=3, kernel_initializer='normal'))
model.add(Dense(24, activation= "relu"))
#model.add(Dense(3, activation= "relu"))
model.add(Dense(1, kernel_initializer='normal',activation="linear"))


adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
gamma = 0.95

model.compile(loss='mean_squared_error', optimizer= adam)


# From https://swaathi.com/2017/04/29/normalizing-data/
def normalizeState(obs,_env):
    l = _env.observation_space.low
    h = _env.observation_space.high
    obs_norm = (obs - l) / (h-l)
    return obs_norm

def normalizeAction(action):
    return (action -0)/(2-0)

def getMaxQvalues(obs,_model,_env):
    obsNorm = normalizeState(obs,_env)
    allActions = np.array([np.append(obsNorm, normalizeAction(0)),
                           np.append(obsNorm, normalizeAction(1)),
                           np.append(obsNorm, normalizeAction(2))])
    q_values = model.predict(allActions)
    #print(q_values)
    return q_values

nrofEpisodes = 2000
epsilon = 0.1
res = np.zeros(shape=(nrofEpisodes,2))
predictedArr = []
actualArr = []

for episode in range(nrofEpisodes):

    #env.render()

    obs, _, done, _ = env.step(env.action_space.sample())
    if (episode % 50) == 0:
        print(episode)
        #epsilon -= 0.2
    while not done:

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = getMaxQvalues(obs, model,env)
            action = np.argmax(q_values)

        new_obs, reward, done, info = env.step(action)

        predicted = np.append(normalizeState(obs,env), normalizeAction(action))
        actual = reward + gamma * np.max(getMaxQvalues(new_obs,model,env))
        predictedArr.append(predicted)
        actualArr.append(actual)
        obs = new_obs

    x = np.vstack(predictedArr)
    y = np.vstack(actualArr)
    trainIdx = np.random.choice(len(y), 128,replace = False)

    #if(len(y)==200):
    #    y[-1] = -200
    #print("%.2f min , %.2f max " % (np.min(x[:,0]),np.max(x[:,0])))
    history = model.fit(x[trainIdx,:], y[trainIdx], epochs=1,verbose=0,batch_size=128)
    #print(history.history,np.min(y),np.max(y))
    env.reset()
    res[episode,:] = [np.min(x[:,0]),np.max(x[:,0])]

plt.plot(res)
plt.show()




