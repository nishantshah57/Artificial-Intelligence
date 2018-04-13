# sudo pip install --upgrade pandas
# sudo pip3 install xlrd
# Pandas version 0.22.0
import time
import pandas as pd
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import pdb

class Map():
    def __init__(self, fileName):
        self.fileName = fileName
        self.goal = None
        self.pit = None
        self.moveCost = None
        self.giveUp = None
        self.trials = None
        self.epsilon = None
        self.gridworld = None

    def gridWorld(self):
        # By default pandas considers 1st row as a header, so need to mention header as none
        df = pd.read_excel(self.fileName,"Sheet1",header=None)
        # df = dataframe
        # .fillna(0) will change all the nan values to 0
        self.showWorld = df.fillna('0').values
        self.showWorld = np.delete(self.showWorld,self.showWorld.shape[0]-1,0)
        self.showWorld = np.delete(self.showWorld,self.showWorld.shape[1]-1,1)
        self.gridworld = df.fillna(self.moveCost).values # df.values turns the dataframe into numpy array
        self.gridworld = np.select([self.gridworld == 'P', self.gridworld == 'G'],[self.pit,self.goal],self.gridworld)
        self.gridworld = np.delete(self.gridworld,self.gridworld.shape[0]-1,0)
        self.gridworld = np.delete(self.gridworld,self.gridworld.shape[1]-1,1)

    def read_arguments(self):
        parser = argparse.ArgumentParser(prog = 'sarsa', description = "Program Name")
        parser.add_argument('Goal', type = float, help = "Goal Reward Value")
        parser.add_argument('Pit', type = float, help = "Pit Penalty Value")
        parser.add_argument('Move', type = float, help = "Move Costs Value")
        parser.add_argument('Giveup', type = float, help = "Give up Costs Value")
        parser.add_argument('Iterations', type = int, help = "How many trials?")
        parser.add_argument('Epsilon', type = float, help = "Epsilon Value")
        args = parser.parse_args()
        self.goal = args.Goal
        self.pit = args.Pit
        self.moveCost = args.Move
        self.giveUp = args.Giveup
        self.trials = args.Iterations
        self.epsilon = args.Epsilon

class Agent(Map):
    def __init__(self,fileName):
        Map.__init__(self,fileName)
        self.actions = 5
        self.qTable = None
        self.alpha = 0.1
        self.gamma = 0.97

    def Qvalue(self):
        # Created a 3d array of qTable which will give the position of agent, qValue and action
        self.qTable = 2*np.ones((self.actions,self.gridworld.shape[0],self.gridworld.shape[1]))
        self.qTable[4,:,:] = -2
        # 0 - UP, 1-RIGHT, 2-DOWN, 3-LEFT, 4-GIVEUP

    def initializeAgent(self):
        State = np.array([np.random.randint(0,self.gridworld.shape[0]), np.random.randint(0,self.gridworld.shape[1])])
        # State = np.random.choice(max(self.gridworld.shape[0]-1,self.gridworld.shape[1]-1),2) # (-1) because array starts with zero we need to subtract 1 from height and width
        # Checking if agent is in pit state or goal state
        while (self.gridworld[State[0],State[1]] == self.pit or self.gridworld[State[0],State[1]] == self.goal):
            # State = np.random.choice(max(self.gridworld.shape[0]-1,self.gridworld.shape[1]-1),2)
            State = np.array([np.random.randint(0,self.gridworld.shape[0]), np.random.randint(0,self.gridworld.shape[1])])
        return State

    def getAction(self, State):
        findAction = []
        for i in range(self.actions):
            findAction.append(self.qTable[i,State[0],State[1]])
        Action = np.argmax(np.array(findAction))
        if random.random() > self.epsilon: # epsilon greedy approach for taking actions
            return Action
        else:
            return np.random.randint(0,5)

    def checkBoundary(self,state):

        if state[0] > self.gridworld.shape[0]-1:
            state[0] -= 1
        elif state[0] < 0:
            state[0] += 1
        elif state[1] > self.gridworld.shape[1]-1:
            state[1] -= 1
        elif state[1] < 0:
            state[1] += 1
        else:
            pass
        return state


    def Movement(self,state,action):
        moveProbs = random.random()
        envProbs = np.array([0.7,0.8,0.9])

        if action == 0: # (Action UP)
            if moveProbs <= envProbs[0]:
                return state + np.array([-1,0]) # Agent decides to move Up
            elif moveProbs > envProbs[0] and moveProbs <= envProbs[1]:
                return state + np.array([0,1]) # Agent decides to move Right
            elif moveProbs > envProbs[1] and moveProbs <= envProbs[2]:
                return state + np.array([0,-1]) # Agent decides to move Left
            else: # Agent decides to move 2 steps forward
                state = state + np.array([-1,0])
                state = self.checkBoundary(state)
                if self.gridworld[state[0],state[1]] == self.pit or self.gridworld[state[0],state[1]] == self.goal:
                    return state
                else:
                    return state + np.array([-1,0])

        elif action == 2:#(Action Down)
            if moveProbs <= envProbs[0]:
                return state + np.array([1,0]) # Agent decides to move Down
            elif moveProbs > envProbs[0] and moveProbs <= envProbs[1]:
                return state + np.array([0,1]) # Agent decides to move Right
            elif moveProbs > envProbs[1] and moveProbs <= envProbs[2]:
                return state + np.array([0,-1]) # Agent decides to move Left
            else: # Agent decides to move 2 steps forward
                state = state + np.array([1,0])
                state = self.checkBoundary(state)
                if self.gridworld[state[0],state[1]] == self.pit or self.gridworld[state[0],state[1]] == self.goal:
                    return state
                    # initialize the agent again
                else:
                    return state + np.array([1,0])

        elif action == 1:#(Action Right)
            if moveProbs <= envProbs[0]:
                return state + np.array([0,1]) # Agent decides to move Right
            elif moveProbs > envProbs[0] and moveProbs <= envProbs[1]:
                return state + np.array([1,0]) # Agent decides to move Right
            elif moveProbs > envProbs[1] and moveProbs <= envProbs[2]:
                return state + np.array([-1,0]) # Agent decides to move Left
            else: # Agent decides to move 2 steps forward
                state = state + np.array([0,1])
                state = self.checkBoundary(state)
                if self.gridworld[state[0],state[1]] == self.pit or self.gridworld[state[0],state[1]] == self.goal:
                    return state
                else:
                    return state + np.array([0,1])

        elif action == 3:#(Action LEFT)
            if moveProbs <= envProbs[0]:
                return state + np.array([0,-1]) # Agent decides to move Forward (Down)
            elif moveProbs > envProbs[0] and moveProbs <= envProbs[1]:
                return state + np.array([1,0]) # Agent decides to move Up
            elif moveProbs > envProbs[1] and moveProbs <= envProbs[2]:
                return state + np.array([-1,0]) # Agent decides to move Down
            else: # Agent decides to move 2 steps forward
                state = state + np.array([0,-1])
                state = self.checkBoundary(state)
                if self.gridworld[state[0],state[1]] == self.pit or self.gridworld[state[0],state[1]] == self.goal:
                    return state
                else:
                    return state + np.array([0,-1])

        elif action == 4: #(Give Up)
            return state

    def reward(self,state,action):
        if(self.gridworld[state[0],state[1]] == self.goal):
            reward = self.goal + self.moveCost
        elif(self.gridworld[state[0],state[1]] == self.pit):
            reward = self.pit + self.moveCost
        elif(action == 4):
            reward = self.giveUp
        else:
            reward = self.moveCost
        return reward

    def sarsa(self,state1,action1,reward,state2,action2):
        self.qTable[action1,state1[0],state1[1]] = self.qTable[action1,state1[0],state1[1]] + self.alpha*(reward + self.gamma*self.qTable[action2,state2[0],state2[1]] - self.qTable[action1,state1[0],state1[1]])

    def getDirections(self):
        learnedPolicy = np.copy(self.gridworld)
        finalQTable = np.copy(self.gridworld)
        for i in range(learnedPolicy.shape[0]):
            for j in range(learnedPolicy.shape[1]):
                State = np.array([i,j])
                if learnedPolicy[i,j] == self.pit:
                    learnedPolicy[i,j] = 'P'
                    finalQTable[i,j] = 'P'
                elif learnedPolicy[i,j] == self.goal:
                    learnedPolicy[i,j] = 'G'
                    finalQTable[i,j] = 'G'
                else:
                    learnedPolicy[i,j] = self.getAction(State)
                    if learnedPolicy[i,j] == 0:
                        learnedPolicy[i,j] = '^'
                        finalQTable[i,j] = (self.qTable[0,i,j])
                    elif learnedPolicy[i,j] == 1:
                        learnedPolicy[i,j] = '>'
                        finalQTable[i,j] = (self.qTable[1,i,j])
                    elif learnedPolicy[i,j] == 2:
                        learnedPolicy[i,j] = 'v'
                        finalQTable[i,j] = (self.qTable[2,i,j])
                    elif learnedPolicy[i,j] == 3:
                        learnedPolicy[i,j] = '<'
                        finalQTable[i,j] = (self.qTable[3,i,j])
                    elif learnedPolicy[i,j] == 4:
                        learnedPolicy[i,j] = 'o'
                        finalQTable[i,j] = (self.qTable[4,i,j])

        return learnedPolicy, finalQTable

    def movingaverage(self,interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')


def main():

    b = Agent('CS 534 map for assignment 41.xlsx')
    b.read_arguments() # Read the arguments from the parser
    b.gridWorld() # Make a gridworld from excel file
    b.Qvalue() # initialize `the Q table
    yRewards = []
    for i in range(b.trials): # Start the number of trials
        totalReward = 0
        count = 0
        State1 = b.initializeAgent() # Initialize the agent on a random cell in the gridworld
        Action1 = b.getAction(State1) # Get Action1
        while True:
            count += 1 # To count if the agent gives up immediately after initialization
            State2 = b.Movement(State1,Action1)
            State2 = b.checkBoundary(State2)
            Action2 = b.getAction(State2) # Get Action2
            reward = b.reward(State2,Action1)
            totalReward += reward
            if(b.gridworld[State2[0],State2[1]] == b.pit or b.gridworld[State2[0],State2[1]] == b.goal or Action1 == 4):
                b.qTable[Action1,State1[0],State1[1]] = b.qTable[Action1,State1[0],State1[1]] + b.alpha*(reward - b.qTable[Action1,State1[0],State1[1]])
                    # print("Reached Here")
                    # break
                break
            else:
                b.sarsa(State1,Action1,reward,State2,Action2)
                State1 = State2
                Action1 = Action2
        # x.append(i)
        yRewards.append(totalReward)
        # print('Trial',i,'-----','Reward',totalReward)
        if i == b.trials-1:
            print("The Original Gridworld is as below: ")
            print(b.showWorld)
            print("\n")
            QTABLE, QVALUES = b.getDirections()
            print("The Directions obtained after the agent has learned the policy: ")
            print(QTABLE)
            # print(np.dtype(QVALUES))
            print("\n")
            # np.set_printoptions(suppress=True)
            # print(QVALUES)

            x = []
            y = []
            for i in range(len(yRewards)):
                x.append(i)
                y.append(np.mean(yRewards[i:i+500]))
                i = i+500

            plt.plot(yRewards)
            plt.plot(x,y)
            plt.xlabel("Number of Iterations")
            plt.ylabel("Rewards Obtained")
            plt.show()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("\n")







# # Q-learning Learn method
# def learn(self, state1, action1, reward, state2):
#     maxqnew = max([self.getQ(state2, a) for a in self.actions])
#     self.learnQ(state1, action1,
#                 reward, reward + self.gamma*maxqnew)
#
# # SARSA learn method
# def learn(self, state1, action1, reward, state2, action2):
#     qnext = self.getQ(state2, action2)
#     self.learnQ(state1, action1,
#                 reward, reward + self.gamma * qnext)
