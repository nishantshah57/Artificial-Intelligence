#!/usr/bin/env python
#import time
from queue import PriorityQueue
import random
from itertools import count
#from timeit import default_timer as timer
#import numpy as np
import time

tie = count() #global

class N_queens:

	def __init__(self,size):
		self.size = size # size of the board(nxn)
		self.state = {} # state of the system represented by an array (index is column and value stored is row)
		self.heuristic = 0 # no of attacking queens
		self.tempstate = {} # dummy attribute to store simulated position

	# Function to randomly assign the initial state of the system
	def random_initial_State(self):
		self.state = {x: random.randint(0,self.size-1)  for x in range(self.size)}
		self.tempstate = dict(self.state)

	# Function to calculate the heuristic
	def heuristic_calculator(self):
		self.heuristic=10 # done to make heuristic admissible
		for i in range(self.size):
			for j in range(i+1,self.size):
				if self.tempstate[i]==self.tempstate[j] or abs(self.tempstate[i]-self.tempstate[j])==abs(i-j):
					self.heuristic+=1
		if(self.heuristic == 10): # if board does not contribute to heuristic set heuristic to 0
			self.heuristic = 0

	# Function to simulate a queen move (to check the heuristic and see if the move is best)
	def moveQueen_simulate(self,Queen_column,direction,steps):
		''' 1-up
		    2-down'''
		self.tempstate = dict(self.state) # reset the tempstate to the actual state
		if direction == 1 and self.tempstate[Queen_column]+steps<self.size:
			self.tempstate[Queen_column] += steps
			return True
		elif direction == 2 and self.tempstate[Queen_column]-steps>=0:
			self.tempstate[Queen_column] -= steps
			return True
		else:
			return False

	# Function to actually move queen and alter board state
	def moveQueen_actual(self):
		self.state = dict(self.tempstate)

class A_star(N_queens):

	def __init__ (self,size):
		N_queens.__init__(self,size)
		self.random_initial_State() # initialize board
		self.initial_state = dict(self.state)
		self.heuristic_calculator()	# calculate heuristic
		#self.heuristic_min = self.heuristic # to store the value of minimum heuristic
		self.decision = [] # stores the value of column of queen and direction of motion
		self.time = time.time()
		self.explored = PriorityQueue(0)
		self.cost = 0
		self.total = self.heuristic
		self.a = list(range(self.size))
		self.expand = []
		self.nodes_expanded = 1
		self.depth = 1

	def cost_calculator(self):
		self.cost = self.total - self.heuristic

	def total_calculator(self,steps):
		self.total = self.cost + self.heuristic + (steps**2) + 10

	def solver(self):
	 	#self.time = time.time() + 10
		while self.heuristic>0:
			#self.heuristic_calculator()
			#print(self.heuristic)
			self.cost_calculator()
			#print("hey")
			#self.decision = []
			for i in self.a:
				for j in [1,2]:
					for k in range(1,self.size):
						#print(self.tempstate,self.state)
						if self.moveQueen_simulate(i,j,k):
							#print(self.tempstate)
							#print(self.cost)
							self.heuristic_calculator()
							#print(self.heuristic)
							self.total_calculator(k)
							#print(self.total)
							#print([self.total,self.tempstate])
							self.explored.put([self.total,next(tie),self.tempstate,i,self.depth])
							#print('yo')
			
			if not self.explored.empty():
				#print('hi')
				self.nodes_expanded += 1
				self.expand = self.explored.get()
				#print(self.tempstate)
				self.state = dict(self.expand[2]);self.total = self.expand[0]
				self.depth = self.expand[4] + 1
				self.tempstate = dict(self.state)
				#print(self.state)
				#print(self.total)
				self.heuristic_calculator()
				#print(self.heuristic)
				self.a = list(range(self.size))
				self.a.remove(self.expand[3])

			if self.heuristic == 0:
				print("Solved")
				self.time = (time.time()-self.time)
				break

				#print(self.heuristic)
			'''if time.time()>self.time or not self.decision:
				print('yo')
				self.decision=[]
				self.restart()
				break
			if self.heuristic == 0:
				print("Solved")
				break
		if self.heuristic == 0:
			print("Solved")
			print("The solved state is:")
			print(problem.state)
			break'''
class Hill_climbing(N_queens):

	def __init__ (self,size):
		N_queens.__init__(self,size)
		self.random_initial_State() # initialize board
		self.heuristic_calculator()	# calculate heuristic
		self.heuristic_min = self.heuristic # to store the value of minimum heuristic
		self.decision = [] # stores the value of column of queen and direction of motion
		self.time = 0
		self.a = list(range(self.size))
		self.nodes_expanded = 1
		self.depth = 1
		self.cost = 0
		self.moves = [dict(self.state)]

	# Function to restart the board
	def restart(self):
		return self.random_initial_State()

	# Iterates trying to solve the board
	def solve_iterate(self):
		self.time = time.time() + 10
		for I in range(10000):
			#print(self.state.values())
			while self.heuristic>0:
				#print("hey")
				self.decision = []
				for i in self.a:
					for j in [1,2]:
						for k in range(1,self.size):
							#print(i,j,k)
							#print(self.tempstate,self.state)
							self.moveQueen_simulate(i,j,k)
							self.heuristic_calculator()
							#print(self.tempstate.values(),self.heuristic)
							if self.heuristic<self.heuristic_min:
                # solves quickly if sidestepping is allowed
								#print(self.heuristic,self.heuristic_min)
								self.heuristic_min = self.heuristic
								self.decision = [i,j,k]
								#print(self.decision)
				#print('Heyyy')
				if self.decision:
					#print('hi')
					self.moveQueen_simulate(self.decision[0],self.decision[1],self.decision[2])
					steps = self.tempstate[self.decision[0]]-self.state[self.decision[0]]
					#print(self.tempstate[i],self.state[i])
					self.cost += 10 + (steps)**2
					self.moveQueen_actual()
					self.moves.append(dict(self.state))
					self.heuristic_calculator()
					self.a = list(range(self.size))
					self.a.remove(self.decision[0])
					self.nodes_expanded += 1
					self.depth += 1
					#print(self.heuristic)
				if not self.decision:
					#print('yo')
					#self.decision=[]
					print("Restart",I+1)
					self.restart()
					self.moves = [dict(self.state)]
					print("The restarted state is:")
					print(self.state)
					self.depth = 1
					self.cost = 0 
					#self.depth = 1
					self.heuristic_calculator()
					self.a = range(self.size)
					self.heuristic_min=self.heuristic
					break
				if self.heuristic == 0:
					print("Solved")
					break
			if self.heuristic == 0:
				print("Solved")
				print("The cost is:", self.cost)
				print("The solved state is:")
				print(self.state)
				self.time = 10-(self.time-time.time())
				print("Sequence of moves is:")
				for i in range(len(self.moves)):
					print("Move",i)
					print(self.moves[i])
				break
			if time.time()>self.time:
				print("Unsolved")
				break
def main():

	n = int(input("Enter 1 for Astar and 2 for Hill Climbing:"))
	m = int(input("Enter the number of queens you what to work with:"))
	#start = timer()
	if m>3:
		if n==1:

			problem=A_star(m)
			print("The initial state of the board is:")
			print(problem.state)
			print("The state is represented by a dictionary. The keys correspond to the column of the board and the value stored is the row in which the queen is placed.")
			problem.solver()
			print("The solved state of the board is:")
			print(problem.tempstate)
			print("The cost to move is:")
			print(problem.total)
			print("Number of nodes expanded", problem.nodes_expanded)
			print("Effective branching factor is", problem.nodes_expanded/problem.depth)
			print("Time elapsed",problem.time)

		elif n==2:
			# m = int(input("Enter the number of queens you want to play with:"))
			problem = Hill_climbing(m)
			print("The initial state of the board is:")
			print(problem.state)
			print("The state is represented by a dictionary. The keys correspond to the column of the board and the value stored is the row in which the queen is placed.")
			problem.solve_iterate()
			print("Number of nodes expanded", problem.nodes_expanded)
			print("Effective branching factor is", problem.nodes_expanded/problem.depth)
			print("Time elapsed",problem.time)

			# end = timer()
			# print("Time elapsed",end - start)
		else:
			print("Invalid")
		#end = timer()
	else:
		print("Can't Solve for number of queens =",m)
	return problem.nodes_expanded

# def main():

if __name__ == "__main__":
	a=main()