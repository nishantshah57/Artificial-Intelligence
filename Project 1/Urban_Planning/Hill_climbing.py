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
tie = count() #global
import time

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
        self.temp_map = []
    
    def get_actual_positions(self,my_map):
        self.i_indices = np.asarray(np.where(my_map == 12)).T
        self.r_indices = np.asarray(np.where(my_map == 13)).T
        self.c_indices = np.asarray(np.where(my_map == 14)).T
        
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

    def move_tile_simulate(self,indices, column, row, type):
        a = self.my_map[row,column]
        if self.given_map[row,column] != 10 and a != 12 and a != 13 and a != 14:
            self.temp_map = self.my_map.copy()
            self.temp_map[indices[0],indices[1]] = 0 
            self.temp_map[row,column] = type
            return True
        else:
            return False

    def move_tile_actual(self,indices, column, row, type):
        self.my_map[indices[0],indices[1]] = 0 
        self.my_map[row,column] = type

class Hill_Climbing(Environment):
    def __init__(self,input):
        Environment.__init__(self,input)
        self.size = 100
        self.my_map = []
        self.time = 0
        self.max_score = 0
        self.solution = []
        self.solution_score =0
        self.time = 0
        self.time_best = 0

    def initialize_map(self):
        self.my_map = np.zeros(self.given_map.shape).flatten()
        self.my_map[0:self.industry] = 12
        self.my_map[self.industry:self.residential+self.industry] = 13
        self.my_map[self.residential+self.industry:self.residential+self.industry+self.commercial] = 14
        random.shuffle(self.my_map)
        self.my_map = self.my_map.reshape(self.given_map.shape)
        self.get_score(self.my_map)
        #print(self.score)
        self.max_score = self.score
        self.decision = []
        self.scores = []

    def Climb_hill(self):
        rows = self.my_map.shape[0]
        columns = self.my_map.shape[1]
        i=0
        self.time = time.time()+10
        while True:
            
            while True:
                self.get_actual_positions(self.my_map)
                self.decision = []
                for row in range(rows):
                    for column in range(columns):
                        for indices in self.i_indices:
                            if self.move_tile_simulate(indices,column,row,12):
                                self.get_score(self.temp_map)
                                #print(self.score)
                                
                                if self.score>self.max_score:
                                    self.max_score = self.score
                                    self.decision = [indices,column,row,12]
                            
                        for indices in self.r_indices:
                            if self.move_tile_simulate(indices,column,row,13):
                                self.get_score(self.temp_map)
                                if self.score>self.max_score:
                                    self.max_score = self.score
                                    self.decision = [indices,column,row,13]
                        
                        for indices in self.c_indices:
                            if self.move_tile_simulate(indices,column,row,14):
                                self.get_score(self.temp_map)
                                if self.score>self.max_score:
                                    self.max_score = self.score
                                    self.decision = [indices,column,row,14]
                #print(self.max_score)
                if self.decision:
                    self.move_tile_actual(self.decision[0],self.decision[1],self.decision[2],self.decision[3])
                    self.get_score(self.my_map)
                    #print(self.decision)
                    #print(self.my_map)
                    if self.score>self.solution_score:
                        self.solution_score = self.score
                        self.solution = [self.score,self.my_map]
                        self.time_best = 10 - (self.time-time.time())
                if not self.decision:
                    self.initialize_map()
                    print("Restart",i+1)
                    i += 1
                    break

            if time.time()>self.time:
                break

if __name__ == '__main__':
    city = Hill_Climbing('sample 3.txt')
    city.initialize_map()
    print(city.score)
    print(city.my_map)
    city.Climb_hill()
    
    city.given_map = city.given_map.astype(str)
    
    for row in range(city.given_map.shape[0]):
        for col in range(city.given_map.shape[1]):
            if city.given_map[row,col] == '10':
                city.given_map[row,col] = 'X'
            elif city.given_map[row,col] == '11':
                city.given_map[row,col] = 'S'

    for i in range(city.my_map.shape[0]):
        for j in range(city.my_map.shape[1]):
            if city.my_map[i,j] == 12:
                city.given_map[i,j] = 'I'
            elif city.my_map[i,j] == 13:
                city.given_map[i,j] = "R"
            elif city.my_map[i,j] == 14:
                city.given_map[i,j] = "C"

    print(city.given_map)

    output = open("final2.txt",'w')
    print("The final score of the best map is: {0}".format(city.solution_score), file = output)
    print("The time at which this score was achieved is:",city.time_best, file=output)
    print("The map of the city:", file = output)
    print(city.given_map, file = output)
    output.close()
    print(city.solution_score)