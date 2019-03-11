

from dqn import DQN
import numpy as np
import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class dqnScorerMountainCar:
    def __init__(self, _dqn, _env):
        self.dqn = _dqn
        self.env = _env
        self.minX =[]
        self.maxX = []
        self.totalReward = []

    # Get the min/max distance in the replay buffer
    def updateResult(self,totalReward):
        replayBuffer = np.vstack(self.dqn.memory)
        minX = np.min(replayBuffer[-199:, 0])
        maxX = np.max(replayBuffer[-199:, 0])
        self.totalReward.append(-totalReward)
        self.minX.append(minX)
        self.maxX.append(maxX)

    def printDistance(self):
        replayBuffer = np.vstack(self.dqn.memory)
        minX = np.min(replayBuffer[:, 0])
        maxX = np.max(replayBuffer[:, 0])
        print("[%.3f , %.3f meanAction = %.1f]" % (minX,maxX,np.mean(replayBuffer[:,2])))

    def plotResults(self,figname):
        plt.figure(figsize=(12,8))
        plt.plot(self.minX,'o')
        plt.plot(self.maxX,'x')
        plt.savefig('figures/xmax_xmin_%s' % figname)

        plt.figure(figsize=(12,8))

        plt.plot(self.totalReward,'x')
        window_size = 50
        window = np.ones(int(window_size)) / float(window_size)
        yav = np.convolve(self.totalReward, window, 'same')
        plt.plot(yav)
        plt.savefig('figures/totalReward_%s' % figname)

        #plt.show()

    def plot_cost_to_go_mountain_car(self,figname,num_tiles=20):

        replayBuffer = np.vstack(self.dqn.memory)
        minx = np.min(replayBuffer[:, 0])
        maxx = np.max(replayBuffer[:, 0])
        miny = np.min(replayBuffer[:, 1])
        maxy = np.max(replayBuffer[:, 1])

        #minx = self.env.observation_space.low[0]
        #maxx = self.env.observation_space.high[0]
        #miny = self.env.observation_space.low[1]
        #maxy = self.env.observation_space.high[1]


        x = np.linspace(minx,maxx , num=num_tiles)
        y = np.linspace(miny,maxy,  num=num_tiles)

        X, Y = np.meshgrid(x, y)
        Z0 = np.apply_along_axis(lambda a:
                                 self.dqn.network.predict(np.array(np.append(a, -1), ndmin=2))[0, 0], 2, np.dstack([X, Y]))

        Z1 = np.apply_along_axis(lambda a:
                                self.dqn.network.predict(np.array(np.append(a,1),ndmin=2))[0,0],2, np.dstack([X, Y]))


        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)#, projection='2d')
        surf = ax.contourf(X, Y, Z1)#, rstride=1, cstride=1,
                               #cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        #ax.set_zlabel('Value')
        title = "Q-value %d" % 2
        ax.set_title(title)

        fig.colorbar(surf)
        plt.savefig('figures/%s_%s' % (title,figname))


        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)  # , projection='2d')
        surf = ax.contourf(X, Y, Z0)  # , rstride=1, cstride=1,
        # cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        # ax.set_zlabel('Value')
        title = "Q-value %d" % 0
        ax.set_title(title)
        fig.colorbar(surf)
        plt.savefig('figures/%s_%s' % (title,figname))

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)  # , projection='2d')
        surf = ax.contourf(X, Y, Z1-Z0)  # , rstride=1, cstride=1,
        # cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        # ax.set_zlabel('Value')
        title = "Diff 2-0"
        ax.set_title(title)
        fig.colorbar(surf)

        plt.savefig('figures/%s_%s' % (title,figname))


        ax.plot(replayBuffer[:,0],replayBuffer[:,1],'x')


        #plt.show()




