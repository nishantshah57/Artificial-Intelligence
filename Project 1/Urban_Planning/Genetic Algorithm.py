# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 01:14:33 2018
@author: Ganesh
"""

import numpy as np
#from sympy.utilities.iterables import multiset_permutations
import random
from itertools import count
from queue import PriorityQueue
tie = count() #global (serves as a tie breaker for elements with same fitness in the priority Queue )
import time
#import matplotlib.pyplot as plt

# to convert the txt input file into a map
class Map:
    def __init__(self,input):
        self.given_map = []
        self.input = input
        self.industry = 0
        self.residential = 0
        self.commercial = 0

    def get_map(self):
        with open(self.input) as f:
    # whenever \n is encountered splitlines will break and take the encountered part aas a string
            my_list = f.read().splitlines()

#my_list = [x.strip() for x in my_list.split(',')]
        self.industry = int(my_list[0])
        self.residential = int(my_list[1])
        self.commercial = int(my_list[2])
        for i in range(3):
            del my_list[0]

        for i in range(len(my_list)):
            self.given_map.append(my_list[i].split(','))

        for i in range(len(self.given_map)):
            for j in range(len(self.given_map[0])):
                if self.given_map[i][j].isdigit():
                    self.given_map[i][j] = int(self.given_map[i][j])
                elif self.given_map[i][j] == 'X':
                    self.given_map[i][j] = 10
                elif self.given_map[i][j] == 'S':
                    self.given_map[i][j] = 11

# coverting into array
        self.given_map = np.array(self.given_map)

class Environment(Map):
    def __init__(self,input):
        Map.__init__(self,input)
        self.get_map()
        self.score = 0
        self.manhattan_check_non_res = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1],
                                [1, 0], [1, 1], [0, -2], [0, 2], [2, 0], [-2, 0]])
        #self.manhattan_check_res = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1],
        #                        [1, 0], [1, 1], [0, -2], [0, 2], [2, 0], [-2, 0]])
        self.manhattan_check_res = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1],
                                [1, 0], [1, 1], [-2, -2],[-2,0],[-2,2],[0,-2],[0,2],[2,-2],[2,0],[2,2],
                                [0, -3], [0, 3], [3, 0], [-3, 0]])
        self.i_indices = []
        self.r_indices = []
        self.c_indices = []
        self.a = []
        
    def get_score(self,my_map):
        self.score = 0
        x_indices = np.asarray(np.where(self.given_map == 10)).T
        s_indices = np.asarray(np.where(self.given_map == 11)).T
        i_indices = np.asarray(np.where(my_map == 12)).T
        r_indices = np.asarray(np.where(my_map == 13)).T
        c_indices = np.asarray(np.where(my_map == 14)).T
        #print(x_indices)
        #print(s_indices)
        #print(i_indices)
        #print(r_indices)
        #print(c_indices)
        for indices in x_indices:
            x_neighbours = self.manhattan_check_non_res + indices
            x_neighbours = (x_neighbours.T[:,x_neighbours.T.min(axis=0)>=0]).T
            for i,j in x_neighbours:
                #print(i,j)
                try:
                    if my_map[i,j] == 12:
                        #print('1')
                        self.score -= 10
                    if my_map[i,j] == 13 or my_map[i,j] == 14:
                        #print('2')
                        self.score -=20
                except IndexError:
                    pass

        for indices in s_indices:
            if my_map[indices[0],indices[1]] != 12 and my_map[indices[0],indices[1]] != 13 and my_map[indices[0],indices[1]] != 14:
                s_neighbours = self.manhattan_check_non_res + indices
                s_neighbours = (s_neighbours.T[:,s_neighbours.T.min(axis=0)>=0]).T
                for i,j in s_neighbours:
                    #print(i,j)
                    try:
                        if my_map[i,j] == 13:
                            #print('3')
                            self.score += 10
                    except IndexError:
                        pass

        for indices in i_indices:
            self.a = indices
            i_neighbours = self.manhattan_check_non_res + indices
            i_neighbours = (i_neighbours.T[:,i_neighbours.T.min(axis=0)>=0]).T    
            for i,j in i_neighbours:
                #print(i,j)
                try:
                    if my_map[i,j] == 12:
                        #print('4')
                        self.score += 3
                except IndexError:
                    pass

            if self.given_map[indices[0],indices[1]] !=11:
                #print('5')
                self.score -= self.given_map[indices[0],indices[1]]
                
        for indices in c_indices:
            #print(indices)
            c_neighbours = self.manhattan_check_non_res + indices
            c_neighbours = (c_neighbours.T[:,c_neighbours.T.min(axis=0)>=0]).T    
            for i,j in c_neighbours:
                #print(i,j)
                try:
                    if my_map[i,j] == 14:
                        #print('6')
                        self.score -= 5
                    
                except IndexError:
                    pass

            if self.given_map[indices[0],indices[1]] !=11:
                #print('5')
                self.score -= self.given_map[indices[0],indices[1]]
            
        for indices in r_indices:
            r_neighbours = self.manhattan_check_res + indices
            #print(r_neighbours)
            r_neighbours = (r_neighbours.T[:,r_neighbours.T.min(axis=0)>=0]).T    
            for i,j in r_neighbours:
                #print(i,j)
                try:
                    if my_map[i,j] == 12:
                        #print('8')
                        self.score -= 5
                    if my_map[i,j] == 14:
                        #print('9')
                        self.score += 5
                except IndexError:
                    pass

            if self.given_map[indices[0],indices[1]] !=11:
                #print('5')
                self.score -= self.given_map[indices[0],indices[1]]
            
class Genetic_Algortithm(Environment):
    def __init__(self,input):
        Environment.__init__(self,input)
        self.size = 100
        self.my_map = []
        self.population = []
        self.min_sorted_population = PriorityQueue(0)
        self.max_sorted_population = PriorityQueue(0)
        self.elite = []
        self.offspring = []
        self.culled = []
        self.culled_index = np.zeros(100)
        self.population_proper = []
        self.time = 0
        self.temp_max = 0
        self.temp_best = []
        self.max_score = float("inf")
        self.best_map = []
        self.scores = []
        self.best_time = 0

    def initialize_map(self):
        self.my_map = np.zeros(self.given_map.shape).flatten()
        self.my_map[0:self.industry] = 12
        self.my_map[self.industry:self.residential+self.industry] = 13
        self.my_map[self.residential+self.industry:self.residential+self.industry+self.commercial] = 14
        random.shuffle(self.my_map)
        
    def populate(self):
        i=0
        while True:
            self.temp_map = self.my_map.copy()
            random.shuffle(self.temp_map)
            a = self.temp_map.reshape(self.given_map.shape)
            b_1 = a==12
            b_2 = a==13
            b_3 = a==14
            if 10 not in self.given_map[b_1] and 10 not in self.given_map[b_2] and 10 not in self.given_map[b_3]:
                self.population.append(a)
                i += 1
            if i == self.size:
                break
    
    '''def populate(self):
        i=0
        for pops in multiset_permutations(self.my_map):
            a = np.array(pops).reshape(self.given_map.shape)
            b_1 = a==12
            b_2 = a==13
            b_3 = a==14
            if 10 not in self.given_map[b_1] and 10 not in self.given_map[b_2] and 10 not in self.given_map[b_3]:
                self.permutations.append(a)
        
        a = random.sample(range(0,len(self.permutations)),self.size)
        for i in a:
            self.population.append(self.permutations[i])'''
    
    def sort_population(self):
        self.min_sorted_population = PriorityQueue(0)
        self.max_sorted_population = PriorityQueue(0)
        for my_map in self.population:
            self.get_score(my_map)
            self.min_sorted_population.put([self.score,next(tie),my_map])
            self.max_sorted_population.put([-self.score,next(tie),my_map])
    
    '''def repopulate(self):
        self.population_proper = []
        self.culled_index = np.zeros(self.size)
        for i in self.culled:
            self.culled_index = self.culled_index + np.array([int(np.array_equal(i,x)) for x in self.population])
            print(sum(self.culled_index))
        for i in range(len(self.culled_index)):
            if self.culled_index[i]==0:
                self.population_proper.append(self.population[i])'''
    
    def repopulate(self):
        self.population_proper = []
        for i in range(self.size-len(self.culled)):
            a = self.min_sorted_population.get()
            self.population_proper.append(a[2])
            
    def culling(self):
        self.culled = []
        for i in range(int(0.1*self.size)):
            b = self.min_sorted_population.get()
            self.culled.append(b[2])
            
    def elitism(self):
        self.elite = []
        for i in range(int(0.1*self.size)):
            a = self.max_sorted_population.get()
            if i==0:
                self.temp_max = a[0]
                self.temp_best = a[2].copy()
            self.elite.append(a[2])

    def crossover(self):
        '''
        Perform mating and produce new offspring
        '''
        self.offspring = []
        numbers = 0
        while True:
            length = len(self.culled)
            int1 = np.random.randint(0,self.size-length)
            int2 = np.random.randint(0,self.size-length)
            if int1 != int2:    
                par_1 = self.population_proper[int1]
                par_2 = self.population_proper[int2]
                
                # random probability
                prob = random.random()

                # if prob is less than 0.35, change industrial tiles
                if prob < 0.35:
                    b_1 = par_1 == 12
                    b_2 = par_2 == 12
                    if 13 not in par_1[b_2] and 14 not in par_2[b_1] and 13 not in par_2[b_1] and 14 not in par_1[b_2]:
                        par_1[b_1] = 0
                        par_2[b_2] = 0
                        par_1[b_2] = 12
                        par_2[b_1] = 12
                        self.offspring.append(par_1)
                        self.offspring.append(par_2)
                        numbers += 2
                
                # if prob is between 0.35 and 0.70, change residential tiles
                elif prob >= 0.35 and prob < 0.7:
                    b_1 = par_1 == 13
                    b_2 = par_2 == 13
                    if 12 not in par_1[b_2] and 14 not in par_2[b_1] and 12 not in par_2[b_1] and 14 not in par_1[b_2]:
                        par_1[b_1] = 0
                        par_2[b_2] = 0
                        par_1[b_2] = 13
                        par_2[b_1] = 13
                        self.offspring.append(par_1)
                        self.offspring.append(par_2)
                        numbers += 2
                
                elif prob >= 0.7 and prob <= 1.0:
                    b_1 = par_1 == 14
                    b_2 = par_2 == 14
                    if 12 not in par_1[b_2] and 13 not in par_2[b_1] and 12 not in par_2[b_1] and 13 not in par_1[b_2]:
                        par_1[b_1] = 0
                        par_2[b_2] = 0
                        par_1[b_2] = 14
                        par_2[b_1] = 14
                        self.offspring.append(par_1)
                        self.offspring.append(par_2)
                        numbers += 2
                
            if numbers == self.size-len(self.elite):
                break
    
    def mutate(self):
        s=0
        while s<20:
            i = np.random.randint(0,len(self.population_proper))
            mutant = self.population_proper[i].copy()
            while True:
                i1 = np.random.randint(0,self.given_map.shape[0])
                i2 = np.random.randint(0,self.given_map.shape[0])
                j1 = np.random.randint(0,self.given_map.shape[1])
                j2 = np.random.randint(0,self.given_map.shape[1])
                if self.given_map[i1,j1]!=10 and self.given_map[i2,j2]!=10:
                    break
            if mutant[i1,j1]!=0 or mutant[i2,j2]!=0:
                a = mutant[i1,j1]
                mutant[i1,j1] = mutant[i2,j2]
                mutant[i2,j2] = a
                self.population_proper[i] = mutant.copy()
                s += 1

    def evolve(self):
        self.initialize_map()
        self.populate()
        self.time = time.time() + 10
        i = 1
        while True:
            print("Generation", i+1)
            self.sort_population()
            self.culling()
            #print(len(self.culled))
            self.elitism()
            if self.temp_max<self.max_score:
                self.max_score = self.temp_max
                self.best_map = self.temp_best.copy()
                self.best_time = 10-(self.time-time.time())
            self.scores.append(-self.max_score)
            print("Max score of population is", -self.max_score)
            #print(len(self.elite))
            self.repopulate()
            self.mutate()
            #print(len(self.population_proper))
            self.crossover()
            #print(len(self.offspring))
            self.offspring.extend(self.elite)
            self.population = self.offspring[:]
            #print(len(self.population))
            i += 1
            if time.time()>self.time:
                break

if __name__ == '__main__':
    city = Genetic_Algortithm('sample 3.txt')
    city.evolve()
    city.given_map = city.given_map.astype(str)
    
    for row in range(city.given_map.shape[0]):
        for col in range(city.given_map.shape[1]):
            if city.given_map[row,col] == '10':
                city.given_map[row,col] = 'X'
            elif city.given_map[row,col] == '11':
                city.given_map[row,col] = 'S'

    for i in range(city.best_map.shape[0]):
        for j in range(city.best_map.shape[1]):
            if city.best_map[i,j] == 12:
                city.given_map[i,j] = 'I'
            elif city.best_map[i,j] == 13:
                city.given_map[i,j] = "R"
            elif city.best_map[i,j] == 14:
                city.given_map[i,j] = "C"

    print(city.given_map)

    output = open("final.txt",'w')
    print("The final score of the best map is: {0}".format(-city.max_score), file = output)
    print("The time at which this score was achieved is:",city.best_time, file=output)
    print("The map of the city:", file = output)
    print(city.given_map, file = output)
    output.close()