# Created on Sun Apr 15 16:40:00 2018
# Author: Chaitanya Pb

import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#%%

class DeepGA:
    
    def __init__(self, env, state_size, action_size, generations, pop_size, elistism, mutation):
        
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        
        self.nn = self.build_model()
        self.num_layers = len(self.nn.layers)
        
        self.population_size = pop_size
        self.mutation = mutation
        self.elitism = elistism
        self.num_generations = generations
        
        self.population = []
        self.pop_fitness = []
        self.children = []
        self.children_fitness = []
    
    def build_model(self):

        # Neural Network to train

        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        return model

    def get_nn_state(self):

        # Return the weights of the NN

        return self.nn.get_weights()

    def set_nn_state(self, weights):

        # Load the weights of the NN

        self.nn.set_weights(weights)

    def noise(self):

        # Generate mutation noise

        return random.triangular(-1, 1)

    def crossover(self, mother, father):
        
        # Crossover
        
        child = []
        crossover_layer = random.randint(1, self.num_layers)
        
        child += mother[0 : 2*crossover_layer]
        child += father[2*crossover_layer : 2*self.num_layers]
        
        return child
    
    def mutate(self, state, strength=0.01):

        # Mutation
        
        for i in range(len(state)):
            for w in np.nditer(state[i], op_flags=['readwrite']):
                w[...] = w + (strength*self.noise())
        
        return state
    
    def get_elite_index(self, num):

        # Get n elite members

        elite = list(np.argpartition(self.pop_fitness, -num)[-num:])
        return elite
    
    def make_children(self):

        # Create the next generation

        self.children = []
        self.children_fitness = []
        
        # Elitism
        num_elite = int(self.elitism * self.population_size)   
        elite = self.get_elite_index(num_elite)
        for i in elite:
            #print("Elitism! Elitism!", i)
            self.children.append(self.population[i])
            self.children_fitness.append(self.pop_fitness[i])
        
        # Mutation
        num_mutations = int(self.mutation * self.population_size)
        mutated = random.sample(range(self.population_size), num_mutations)
        
        for i in mutated:
            #print("Mutation! Mutation!", i)
            child = self.mutate(self.population[i], strength=0.05)
            self.children.append(child)
            self.children_fitness.append(self.fitness(child))
        
        # Crossover
        num_crossovers = self.population_size - (num_elite + num_mutations)
        
        for i in range(num_crossovers):
            #print("Crossover! Crossover!", i)
            mom, dad = self.select_parents()
            child = self.crossover(mom, dad)
            self.children.append(child)
            self.children_fitness.append(self.fitness(child))
        
        return 0
    
    def populate_random(self):
        
        # Generate random population

        init_state = self.get_nn_state()
        
        self.population.append(init_state)
        self.pop_fitness.append(self.fitness(init_state))

        for i in range(self.population_size-1):
            
            new_state = self.mutate(init_state, strength=0.5)
            
            self.population.append(new_state)
            self.pop_fitness.append(self.fitness(new_state))

        return 0
    
    def select_parents(self):

        # Randomly select two parents

        #elite = self.get_elite_index(50)
        
        parents = random.sample(self.population, 2)
        return parents[0], parents[1]
    
    def fitness(self, state):

        # Fitness function

        self.set_nn_state(state)
        fitness_value = self.run_episode()

        return fitness_value

    def run_episode(self):

        rewards = []
        done = False

        obs = self.env.reset()
        obs = np.reshape(obs,[1,self.state_size])
        
        # print(obs)
        while not done:
            action = self.nn.predict(obs)
            obs, reward, done, _ = self.env.step(action)
            obs = np.reshape(obs,[1,self.state_size])
            rewards += [reward]

        return sum(rewards)

    def run(self):

        ''' Run Deep GA '''

        best_fitness = []
        mean_fitness = []

        # Random initialization
        print("Generation 0")
        self.populate_random()

        # Print best fitness of the generation
        best = self.get_elite_index(1)
        print("Best in this generation =", self.pop_fitness[best[0]])
        print("Average in this generation =", np.mean(self.pop_fitness))

        # GA for many generations
        for g in range(self.num_generations):

            print("Generation", g+1)
            
            # New generation
            self.make_children()

            # Population <-- Children
            self.population = self.children
            self.pop_fitness = self.children_fitness

            # Print best fitness of the generation
            best = self.get_elite_index(1)
            print("Best in this generation =", self.pop_fitness[best[0]])
            print("Average in this generation =", np.mean(self.pop_fitness))

            best_fitness += [self.pop_fitness[best[0]]]
            mean_fitness += [np.mean(self.pop_fitness)]
        
        plt.plot(best_fitness)
        plt.plot(mean_fitness)
        plt.show()

        return 0
            
# ------------------------ Main Code ------------------------

if __name__ == "__main__":

    env = gym.make('Walker2d-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    pop_size = 1000
    elitism = 0.1
    mutation = 0.5
    generations = 200

    dga = DeepGA(env, state_size, action_size, generations, pop_size, elitism, mutation)
    dga.run()
    