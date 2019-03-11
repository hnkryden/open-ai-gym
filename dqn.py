from collections import deque
import random
import numpy as np
import keras
class DQN:
    def __init__(self,_model,_gamma,_memorysize, replaysize, _env):
        self.memorySize = _memorysize
        self.memory = deque(maxlen=self.memorySize)
        self.network = _model
        self.targetNetwork = keras.models.clone_model(self.network)
        self.epsilon = 1
        self.gamma = _gamma
        self.replaySize = replaysize
        self.env = _env
        self.modelHistory = []
        self.episode = 0
        self.targetNetworkUpdate = 10
        #self.updateTargetNetwork()

    def add(self,action,obs,new_obs,reward,dqn=False):
        obsNorm     = self.normalizeObs(obs)
        new_obsNorm = self.normalizeObs(new_obs)
        actionNorm  = self.normalizeAction(action)
        actions = self.getActionSet()
        q_val = []
        # Double DQN implementation
        for a in actions:
            tmp = np.array(np.append(new_obsNorm,a),ndmin=2)
            q_val.append(self.network.predict(tmp))

        maxQ = max(q_val)

        if(dqn):
            bestAction = np.argmax(q_val)
            actionNormTarget = self.getActionSet()[bestAction]
            tmp = np.array(np.append(new_obsNorm, actionNormTarget), ndmin=2)
            maxQ = self.targetNetwork.predict(tmp)[0,0]


        if(reward<0):
            totalReward = reward + self.gamma * maxQ
        else:
            totalReward = reward
        sample = np.append(obsNorm,actionNorm)
        self.memory.append(np.append(sample,totalReward))

    def getActionSet(self):
        actionSet = [0,2]
        return self.normalizeAction(np.array(actionSet))

    # From https://swaathi.com/2017/04/29/normalizing-data/

    def normalizeObs(self,obs):
        l = self.env.observation_space.low
        h = self.env.observation_space.high
        obs_norm = (obs - l) / (h - l)
        #return obs_norm
        return obs

    # assume we only use 0,2 action
    def normalizeAction(self, action):
        return action - 1

    def updateTargetNetwork(self):
        self.targetNetwork.set_weights(self.network.get_weights())
        #print("update target network")

    def sample(self):
        return random.sample(self.memory, self.replaySize)

    def replay(self):
        self.episode += 1
        if(self.episode > 500):#500): #epsilon>0.1):
            self.epsilon = 0.1
        if((self.episode % self.targetNetworkUpdate) == 0):
            self.updateTargetNetwork()
        #if(self.episode > 2800):#500): #epsilon>0.1):
        #    self.epsilon = 0.01
        samples = self.sample()
        replay = np.vstack(samples)
        x = replay[:, 0:3]
        y = replay[:, 3]
        output = self.network.fit(x, y, epochs=10, verbose=0, batch_size=32)
        self.modelHistory.append(output.history)
        return

    def action(self,obs):
        obsNorm = self.normalizeObs(obs)
        if self.epsilon > np.random.rand():
            action = 0 if np.random.rand() < 0.5 else 2
        else:
            q_val = []
            for a in self.getActionSet():
                tmp = np.array(np.append(obsNorm, a), ndmin=2)
                qtmp = self.network.predict(tmp)
                q_val.append(qtmp[0,0])
            best = np.argmax(q_val)
            actions = [0,2]
            action = actions[int(best)]
        return action
