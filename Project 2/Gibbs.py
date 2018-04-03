from __future__ import print_function
import random
from collections import Counter
import argparse
import time
import matplotlib.pyplot as plt

class Parser(object):
	def __init__(self):
		self.interestNode = 'None'  																					# stores the string for the query node
		self.evidenceNode = []																							# list of the evidence nodes
		self.evidenceState = []																							# list of states the evidence node is at
		self.iterations = 1 																							# number of iterations
		self.discard_values = 0 																						# number of values to discard

	def read_arguments(self):
		parser = argparse.ArgumentParser(prog = 'gsss', description = "Pass in all the arguments")
		parser.add_argument('QueryNode', type = str, help = "Enter the node to be queried")
		parser.add_argument('Evidence', nargs = '+', help = 'Enter the evidence nodes with its value')
		parser.add_argument('-u', type = int, help = "Number of Iterations")
		parser.add_argument('-d', type = int, default = 0, help = "Number of initial samples to drop")
		args = parser.parse_args()
		self.interestNode = args.QueryNode
		temp = args.Evidence
		self.evidenceNode = [i.split('=', -1)[0] for i in temp]
		self.evidenceState = [i.split('=', -1)[1] for i in temp]
		self.iterations = args.u
		self.discard_values = args.d

class Network(Parser):
	def __init__(self):																									# This is the given network
		Parser.__init__(self)
		self.graph = {'amenities'	 : ['location'],
					  'neighborhood' : ['location', 'children'],
					  'location'	 : ['age', 'price'],
					  'children'	 : ['schools'],
					  'age'			 : ['price'],
					  'schools'		 : ['price'],
					  'size'		 : ['price']}

class CPT(Network):																										# This is the CPT table given
	def __init__(self):
		Network.__init__(self)

	def Cpt_amenities(self,condition):
		p_amenities = {}
		p_amenities = {'lots':0.3, 'little':0.7}
		return  p_amenities[condition]

	def Cpt_neighborhood(self,condition):
		p_neighborhood = {}
		p_neighborhood = {'bad':0.4, 'good':0.6}
		return p_neighborhood[condition]

	def Cpt_location(self,condition,amenities,neighborhood):
		p_location= {}
		if amenities == 'lots' and neighborhood == 'bad':
			p_location = {'good':0.3, 'bad':0.4, 'ugly':0.3}
		elif amenities == 'lots' and neighborhood == 'good':
			p_location = {'good':0.8, 'bad':0.15, 'ugly':0.05}
		elif amenities == 'little' and neighborhood == 'bad':
			p_location = {'good':0.2, 'bad':0.4, 'ugly':0.4}
		elif amenities == 'little' and neighborhood == 'good':
			p_location = {'good':0.5, 'bad':0.35, 'ugly':0.15}
		return p_location[condition]

	def Cpt_children(self,condition,neighborhood):
		p_children ={}
		if neighborhood == 'bad':
			p_children = {'bad':0.6, 'good':0.4}
		elif neighborhood == 'good':
			p_children = {'bad':0.3, 'good':0.7}
		return p_children[condition]

	def Cpt_size(self,condition):
		p_size = {}
		p_size = {'small':0.33 , 'medium':0.34, 'large':0.33}
		return p_size[condition]

	def Cpt_schools(self,condition,children):
		p_schools = {}
		if children == 'bad':
			# print('hi')
			p_schools = {'bad':0.7, 'good':0.3}
		#	return p_schools[conditionools]
		elif children == 'good':
			p_schools = {'bad':0.2, 'good':0.8}
		return p_schools[condition]

	def Cpt_age(self,condition,location):
		p_age = {}
		if location =='good':
			p_age = {'old':0.3, 'new':0.7}
		elif location =='bad':
			p_age = {'old':0.6, 'new':0.4}
		elif location =='ugly':
			p_age = {'old':0.9, 'new':0.1}
		return p_age[condition]

	def Cpt_price(self,condition,location,age,schools,size):
		p_price = {}
		# 40
		if location == 'good' and age == 'old' and schools == 'bad' and size =='small':
			p_price = {'cheap':0.5 ,'ok':0.4, 'expensive':0.1}
		# 41
		elif location == 'good' and age == 'old' and schools == 'bad' and size =='medium':
			p_price = {'cheap':0.4 ,'ok':0.45, 'expensive':0.15}
		# 42
		elif location == 'good' and age == 'old' and schools == 'bad' and size =='large':
			p_price = {'cheap':0.35 ,'ok':0.45, 'expensive':0.2}
		# 43
		elif location == 'good' and age == 'old' and schools == 'good' and size =='small':
			p_price = {'cheap':0.4 ,'ok':0.3, 'expensive':0.3}
			# 44
		elif location == 'good' and age == 'old' and schools == 'good' and size =='medium':
			p_price = {'cheap':0.35 ,'ok':0.3, 'expensive':0.35}
			# 45
		elif location == 'good' and age == 'old' and schools == 'good' and size =='large':
			p_price = {'cheap':0.3 ,'ok':0.25, 'expensive':0.45}
			# 46
		elif location == 'good' and age == 'new' and schools == 'bad' and size =='small':
			p_price = {'cheap':0.45 ,'ok':0.4, 'expensive':0.15}
			# 47
		elif location == 'good' and age == 'new' and schools == 'bad' and size =='medium':
			p_price = {'cheap':0.4 ,'ok':0.45, 'expensive':0.15}
			# 48
		elif location == 'good' and age == 'new' and schools == 'bad' and size =='large':
			p_price = {'cheap':0.35 ,'ok':0.45, 'expensive':0.2}
			# 49
		elif location == 'good' and age == 'new' and schools == 'good' and size =='small':
			p_price = {'cheap':0.25 ,'ok':0.3, 'expensive':0.45}
			# 50
		elif location == 'good' and age == 'new' and schools == 'good' and size =='medium':
			p_price = {'cheap':0.2 ,'ok':0.25, 'expensive':0.55}
			# 51
		elif location == 'good' and age == 'new' and schools == 'good' and size =='large':
			p_price = {'cheap':0.1 ,'ok':0.2, 'expensive':0.7}
			# 52
		elif location == 'bad' and age == 'old' and schools == 'bad' and size =='small':
			p_price = {'cheap':0.7 ,'ok':0.299, 'expensive':0.001}
			# 53
		elif location == 'bad' and age == 'old' and schools == 'bad' and size =='medium':
			p_price = {'cheap':0.65 ,'ok':0.33, 'expensive':0.02}
			# 54
		elif location == 'bad' and age == 'old' and schools == 'bad' and size =='large':
			p_price = {'cheap':0.65 ,'ok':0.32, 'expensive':0.03}
			# 55
		elif location == 'bad' and age == 'old' and schools == 'good' and size =='small':
			p_price = {'cheap':0.55 ,'ok':0.35, 'expensive':0.1}
			# 56
		elif location == 'bad' and age == 'old' and schools == 'good' and size =='medium':
			p_price = {'cheap':0.5 ,'ok':0.35, 'expensive':0.15}
			# 57
		elif location == 'bad' and age == 'old' and schools == 'good' and size =='large':
			p_price = {'cheap':0.45 ,'ok':0.4, 'expensive':0.15}
			# 58
		elif location == 'bad' and age == 'new' and schools == 'bad' and size =='small':
			p_price = {'cheap':0.6 ,'ok':0.35, 'expensive':0.05}
			# 59
		elif location == 'bad' and age == 'new' and schools == 'bad' and size =='medium':
			p_price = {'cheap':0.55 ,'ok':0.35, 'expensive':0.1}
			# 60
		elif location == 'bad' and age == 'new' and schools == 'bad' and size =='large':
			p_price = {'cheap':0.5 ,'ok':0.4, 'expensive':0.1}
			# 61
		elif location == 'bad' and age == 'new' and schools == 'good' and size =='small':
			p_price = {'cheap':0.4 ,'ok':0.4, 'expensive':0.2}
			# 62
		elif location == 'bad' and age == 'new' and schools == 'good' and size =='medium':
			p_price = {'cheap':0.3 ,'ok':0.4, 'expensive':0.3}
			# 63
		elif location == 'bad' and age == 'new' and schools == 'good' and size =='large':
			p_price = {'cheap':0.3 ,'ok':0.3, 'expensive':0.4}
			# 64
		elif location == 'ugly' and age == 'old' and schools == 'bad' and size =='small':
			p_price = {'cheap':0.8 ,'ok':0.1999, 'expensive':0.0001}
			# 65
		elif location == 'ugly' and age == 'old' and schools == 'bad' and size =='medium':
			p_price = {'cheap':0.75 ,'ok':0.24, 'expensive':0.01}
			# 66
		elif location == 'ugly' and age == 'old' and schools == 'bad' and size =='large':
			p_price = {'cheap':0.75 ,'ok':0.23, 'expensive':0.02}
			# 67
		elif location == 'ugly' and age == 'old' and schools == 'good' and size =='small':
			p_price = {'cheap':0.65 ,'ok':0.3, 'expensive':0.05}
			# 68
		elif location == 'ugly' and age == 'old' and schools == 'good' and size =='medium':
			p_price = {'cheap':0.6 ,'ok':0.33, 'expensive':0.07}
			# 69
		elif location == 'ugly' and age == 'old' and schools == 'good' and size =='large':
			p_price = {'cheap':0.55 ,'ok':0.37, 'expensive':0.08}
			# 70
		elif location == 'ugly' and age == 'new' and schools == 'bad' and size =='small':
			p_price = {'cheap':0.7 ,'ok':0.27, 'expensive':0.03}
			# 71
		elif location == 'ugly' and age == 'new' and schools == 'bad' and size =='medium':
			p_price = {'cheap':0.64 ,'ok':0.3, 'expensive':0.06}
			# 72
		elif location == 'ugly' and age == 'new' and schools == 'bad' and size =='large':
			p_price = {'cheap':0.61 ,'ok':0.32, 'expensive':0.07}
			# 73
		elif location == 'ugly' and age == 'new' and schools == 'good' and size =='small':
			p_price = {'cheap':0.48 ,'ok':0.42, 'expensive':0.1}
			# 74
		elif location == 'ugly' and age == 'new' and schools == 'good' and size =='medium':
			p_price = {'cheap':0.41 ,'ok':0.39, 'expensive':0.2}
			# 75
		elif location == 'ugly' and age == 'new' and schools == 'good' and size =='large':
			p_price = {'cheap':0.37 ,'ok':0.33, 'expensive':0.3}
		return p_price[condition]

class gibbsSampling(CPT):
	def __init__(self):
		CPT.__init__(self)
		self.numberOfNodes		= 8																						# number of nodes in the network
		self.markovBlanket 		= []																					# list to store the markov blanket of query node
		self.markovBlanket_temp	= []																					# stores markov blanket of sampled node(random)
		#self.evidenceNode		= self.evidence
		#self.evidenceState 	= state
		#self.interestNode 		= self.Query
		self.temp_interest 		= None 																					# temporary variable
		self.state 				= ['amenities','schools','age','neighborhood','children','location','price','size']		# contains the list of all nodes
		self.amenities_count 	= Counter()																				# counter to count number of times a value is assigned to amenities
		self.schools_count		= Counter()																				# counter to count number of times a value is assigned to schools
		self.neighborhood_count = Counter()																				# counter to count number of times a value is assigned to neighborhood
		self.children_count 	= Counter()																				# counter to count number of times a value is assigned to children
		self.location_count		= Counter()																				# counter to count number of times a value is assigned to location
		self.age_count 			= Counter()																				# counter to count number of times a value is assigned to age
		self.price_count		= Counter()																				# counter to count number of times a value is assigned to price
		self.size_count			= Counter()																				# counter to count number of times a value is assigned to size
		self.sampleSpace		= {'amenities'	   : ['lots','little'],													# contains the values/states each node can take
						   		   'neighborhood' : ['good','bad'],
						   		   'location'	   : ['good','bad','ugly'],
						   		   'children'	   : ['good','bad'],
						   		   'age'	   	   : ['old','new'],
						   		   'schools'	   : ['good','bad'],
						   		   'size'	   	   : ['small','medium','large'],
						   		   'price'	 	   : ['cheap','ok','expensive']}

		# self.plot_2			= {}
		self.blop			 	= 0
		self.plot 				= {}
		self.amenitiesProb 		= {}
		self.schoolsProb		= {}
		self.neighborhoodProb 	= {}
		self.childrenProb		= {}
		self.locationProb		= {}
		self.ageProb 			= {}
		self.priceProb			= {}
		self.sizeProb			= {}
		self.count  			= 1																						# counts number of iterations
		self.nonEvidence 		= []																					# contains the list of nodes other than evidence
		#self.amenities 		= None
		#self.schools			= None
		#self.neighborhood 		= None
		#self.children 			= None
		#self.location			= None
		#self.age 				= None
		#self.price				= None
		#self.size				= None

	def getParents(self,node):																							# gets the parents of the query node
		for parents,child in self.graph.items():
			try:
				if child.index(node)+1:
					#print(parents)
					if parents not in self.markovBlanket_temp and parents!=self.temp_interest:
						self.markovBlanket_temp.append(parents)
			except ValueError:
				pass

	def getChildren(self,node):																							# gets children of the query node and its parents
		try:
			children = self.graph[node]
			#print(children)
			self.markovBlanket_temp.extend(children)
			for child in children:
				self.getParents(child)
		except KeyError:
			pass

	def getMarkovBlanket(self,node):																					# gets the Markov Blanket
		self.temp_interest = node
		self.markovBlanket_temp = []
		self.getParents(node)
		self.getChildren(node)
		if node == self.interestNode:
			self.markovBlanket = self.markovBlanket_temp[:]

	def randomize_network(self):																					 	# randomly assigns the network except for the evidence node
		# print(self.state)
		# list(set(self.state).difference(set(self.evidenceNode)))
		# print(self.state)
		self.state = list(set(self.state) - set(self.evidenceNode))
		# print(self.state)
		self.nonEvidence = list(self.state)
		# print(self.state)
		for i in range(len(self.evidenceNode)):
			setattr(self,self.evidenceNode[i],self.evidenceState[i])
		for iteration in range(len(self.state)):
			a = self.state.pop()
			# print(a)
			# print(self.state)
			coinToss = random.random()

			if a == 'neighborhood' or a == 'schools' or a == 'children':
				if coinToss <= 0.5:
					setattr(self,a,'good')
				else:
					setattr(self,a,'bad')

			elif a == 'amenities':
				if coinToss <= 0.5:
					setattr(self,a,'lots')
				else:
					setattr(self,a,'little')

			elif a == 'location':
				if coinToss <= 0.33:
					setattr(self,a,'good')
				elif coinToss>0.33 and coinToss <= 0.67:
					setattr(self,a,'bad')
				else:
					setattr(self,a,'ugly')

			elif a == 'size':
				if coinToss <= 0.33:
					setattr(self,a,'small')
				elif coinToss>0.33 and coinToss <= 0.67:
					setattr(self,a,'medium')
				else:
					setattr(self,a,'large')

			elif a == 'price':
				if coinToss <= 0.33:
					setattr(self,a,'cheap')
				elif coinToss>0.33 and coinToss <= 0.67:
					setattr(self,a,'ok')
				else:
					setattr(self,a,'expensive')

			elif a == 'age':
				if coinToss <= 0.5:
					setattr(self,a,'old')
				else:
					setattr(self,a,'new')

		# print(self.amenities)
		# print(self.schools)
		# print(self.neighborhood)
		# print(self.children)
		# print(self.location)
		# print(self.age)
		# print(self.price)
		# print(self.size)

	def Probability(self,node):																							# returns the probability of a node given its parents
		if node == 'size':
			return self.Cpt_size(self.size)

		if node == 'amenities':
			return self.Cpt_amenities(self.amenities)

		if node == 'neighborhood':
			return self.Cpt_neighborhood(self.neighborhood)

		if node == 'location':
			return self.Cpt_location(self.location,self.amenities,self.neighborhood)

		if node == 'children':
			return self.Cpt_children(self.children,self.neighborhood)

		if node == 'age':
			return self.Cpt_age(self.age,self.location)

		if node == 'schools':
			#print(self.schools,self.children)
			return self.Cpt_schools(self.schools,self.children)

		if node == 'price':
			return self.Cpt_price(self.price,self.location,self.age,self.schools,self.size)

	def getProbabilityDistribution(self,node):																			# gives the probability distribution for the node being sampled
		a = {}																											# this probability is used to assign value/state to the node by rolling a die
		p = 1
		for i in self.sampleSpace[node]:
			setattr(self,node,i)
			if node == self.interestNode and self.blop<=len(self.sampleSpace[self.interestNode]):
				self.plot[i] = []
				# self.plot_2[i] = []
				self.blop	+= 1
			#print(1,i)
			#print(2,self.children)
			p = self.Probability(node)
			try:
				for j in self.graph[node]:
					#print(3,j)
					#print(4,self.schools)
					p = p*self.Probability(j)
			except KeyError:
				pass

			a[i] = p
		a = {i: p / sum(a.values()) for i, p in a.items()}
		setattr(self,node+'Prob',dict(a))

	def setNode(self,node):																								# sets the node based on the probabiltiy distribution computed in the previous step
		a = dict(getattr(self,node+'Prob'))
		k = 0
		coinToss = random.random()
		count = 0
		for i,j in a.items():
			if coinToss>k and coinToss<=j+k:
				setattr(self,node,i)
				if self.count>self.discard_values:
					b = Counter(getattr(self,node+'_count'))
					b.update({i:1})
					setattr(self,node+'_count',Counter(b))
					a = dict(getattr(self,self.interestNode+'_count'))
					a = {x: y / sum(a.values(),0.0) for x, y in a.items()}
					{self.plot[x].append(y) for x,y in a.items()}
				# else:
				# 	if getattr(self,node+'_count') or count <1:
				# 		b_2 = Counter(getattr(self,node+'_count'))
				# 		count = 1
				# 	else:
				# 	# print(b_2)
				# 		b_2 = Counter(getattr(self,node+'_count_2'))
				# 	# b_2.update({i:1})
				# 	print(dict(b_2))
				# 	print(node)
				# 	setattr(self,node+'_count_2',Counter(b_2))
				# 	a_2 = dict(getattr(self,self.interestNode+'_count_2'))
				# 	print(getattr(self,self.interestNode+'_count_2'))
				# 	a_2 = {x: y / sum(a_2.values(),0.0) for x, y in a_2.items()}
				# 	{self.plot_2[x].append(y) for x,y in a_2.items()}
			k = j

	def solver(self):
		self.randomize_network()
		for i in range(self.iterations):
			# print(self.state)
			node = random.choice(self.nonEvidence)
			self.getMarkovBlanket(node)
			self.getProbabilityDistribution(node)
			self.setNode(node)
			self.count += 1
		a = dict(getattr(self,self.interestNode+'_count'))
		# print(a)
		a = {i: p / sum(a.values(),0.0) for i, p in a.items()}
		for i,j in a.items():
			print("P("+str(self.interestNode)+"="+str(i)+") =",j)
		# print(self.count)

if __name__ == "__main__":
	a = gibbsSampling()
	a.read_arguments()
	# print(a.interestNode)
	# a.getMarkovBlanket(a.interestNode)
	# print(a.markovBlanket,a.markovBlanket_temp)
	# a.getMarkovBlanket('location')
	# print(a.markovBlanket,a.markovBlanket_temp)
	# a.randomize_network()
	# print(a.evidenceNode)
	# print(a.nonEvidence)
	# a.getProbabilityDistribution(a.interestNode)
	# print((getattr(a,a.interestNode+'Prob')))
	# print(a.amenitiesProb)
	times = time.time()
	a.solver()
	times = time.time() - times
	# print(a.plot)
	print("Time taken for",a.iterations,"iterations is",times,"s")
	for i,j in a.plot.items():
		plt.plot(j,label=i)
	plt.xlabel("Number of iterations")
	plt.ylabel("Probability")
	plt.legend()
	plt.show()
	# for i,j in a.plot.items():
	# 	plt.plot(j,label='Discarded 1000')
	# 	plt.plot(a.plot_2[i],label='Undiscarded')
	# 	plt.xlabel("Number of iterations")
	# 	plt.ylabel("Probability")
	# 	plt.legend()
	# 	plt.show()