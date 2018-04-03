from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import argparse

class data(object):
	def __init__(self):
		self.data 	    		= None
		self.file_name  		= None
		self.mean 	    		= None
		self.covariance 		= None
		self.numberOfClusters	= None
		self.clusterProbability = None
		self.dataGivenCluster	= None
		self.clusterGivenData	= None
		self.labels				= None

	def readData(self):
		self.data = np.genfromtxt(self.file_name,delimiter=',')

	def setClusterProbabilities(self):
		self.clusterProbability   = np.ones(self.numberOfClusters)/self.numberOfClusters
	
	def setMean(self):
		temp = random.sample(range(self.data.shape[0]),self.numberOfClusters)
		self.mean = np.asarray([self.data[i,:] for i in temp])

	def setCovariance(self):
		a = np.abs(np.amin(self.data) - np.amax(self.data))
		self.covariance = np.asarray([a*np.identity(self.data.shape[1]) for i in range(self.numberOfClusters)])

	def initialize(self):
		self.setMean()
		self.setCovariance()

	def read_arguments(self):
		parser = argparse.ArgumentParser(prog = 'em', description = "Pass in the data file name and the number of clusters")
		parser.add_argument('dataFile', type = str, help = "Enter the data filename")
		parser.add_argument('Clusters', nargs = '+', help = 'Enter number of clusters')
		args = parser.parse_args()
		self.file_name = args.dataFile
		temp_clusters = args.Clusters
		if temp_clusters[-1].isdigit():
		    self.numberOfClusters = int(temp_clusters[-1])
		else:
		    self.numberOfClusters = temp_clusters[-1]

class multivariateGaussian(object):	
	def getprobability(self,x,mean,sd):
		a 	= np.linalg.solve(sd,(x-mean).T).T
		b 	= (x-mean)
		pdf = np.exp(-0.5*np.sum(b*a,axis=1))/(((2*np.pi**x.shape[1])*np.linalg.det(sd))**0.5)
		return pdf

class expectation(multivariateGaussian,data):
	def __init__(self):
		data.__init__(self)

	def expect(self):
		self.dataGivenCluster = []
		for i in range(self.numberOfClusters):
			self.dataGivenCluster.append(self.getprobability(self.data,self.mean[i],self.covariance[i]))
		self.dataGivenCluster = np.asarray(self.dataGivenCluster)
		self.clusterGivenData = self.dataGivenCluster.T*self.clusterProbability
		self.clusterGivenData = self.clusterGivenData/np.sum(self.clusterGivenData,axis=1)[:,np.newaxis]
		
class maximization(multivariateGaussian,data):
	def __init__(self):
		data.__init__(self)
	
	def maximize(self):
		self.clusterProbability = np.sum(self.clusterGivenData,axis=0)/self.data.shape[0]
		self.mean = np.dot(self.clusterGivenData.T,self.data)/np.sum(self.clusterGivenData,axis=0)[:,np.newaxis]
		self.covariance = [np.dot(((self.clusterGivenData[:,i].T*(self.data-self.mean[i,:]).T)),(self.data-self.mean[i,:]))/np.sum(self.clusterGivenData,axis=0)[i] for i in range(self.numberOfClusters)]

class EM(expectation,maximization):
	def __init__(self):
		expectation.__init__(self)
		maximization.__init__(self)
		# self.file_name			  = fileName
		# self.numberOfClusters	  = numberOfClusters
		# self.clusterProbability   = np.ones(numberOfClusters)/numberOfClusters
		self.dataGivenCluster	  = []
		self.clusterGivenData	  = []
		self.logLikelihood	      = [1]
		self.temp_covariance      = None
		self.temp_mean			  = None
		self.temp_cgd			  = None
		self.temp_logLikelihood	  = None
		
	def getlogLikelihood(self):
		a = self.dataGivenCluster==0
		self.dataGivenCluster[a] = 1e-100
		self.logLikelihood.append(int(np.sum(np.log(np.sum((self.clusterProbability)*(self.dataGivenCluster).T,axis=1)))))

	def findCluster(self):
		self.initialize()
		self.logLikelihood = [1]
		temp = 0
		epsilon = 1e-4
		i=0
		while abs(temp-self.logLikelihood[-1])>epsilon:
			temp = self.logLikelihood[-1]
			self.expect()
			self.maximize()
			self.getlogLikelihood()
			i+=1
		# print(self.logLikelihood[-1])
		return self.logLikelihood[-1]

	def findBestCluster(self):
		new = -float('Inf')
		for i in range(20):
			# print("Restart ",i)
			prev = new
			new  = self.findCluster() 
			if new>prev:
				self.temp_mean = np.copy(self.mean)
				self.temp_covariance = np.copy(self.covariance)
				self.temp_cgd	    = np.copy(self.clusterGivenData)
				self.temp_logLikelihood = np.copy(self.logLikelihood)

	def getBestParameters(self):
		setattr(EM,'bestMean',np.copy(self.temp_mean))
		setattr(EM,'bestCovariance',np.copy(self.temp_covariance))
		setattr(EM,'bestClusterGivenData',np.copy(self.temp_cgd))
		setattr(EM,'bestLogLikelihood',np.copy(self.temp_logLikelihood))
	
	def BIC(self):
		return 2*self.logLikelihood[-1] - ((self.data.shape[1]*(self.data.shape[1]+1)/2+self.data.shape[1])*i+i-1)*np.log(self.data.shape[0])

	def visualize(self):
		b = np.argmax(self.temp_cgd,axis=1)
		for i in range(self.data.shape[0]):
			for j in range(self.numberOfClusters):
				if b[i]==j:
					plt.plot(self.data[i,0],self.data[i,1],'C'+str(j)+'.')
		plt.show()
		plt.plot(self.temp_logLikelihood[1:])
		plt.xlabel('number of iterations')
		plt.ylabel('Log-likelihood')
		plt.show()
			
	def BICvisualize(self):
		b = np.argmax(self.bestClusterGivenData,axis=1)
		for i in range(self.data.shape[0]):
			for j in range(self.numberOfClusters):
				if b[i]==j:
					plt.plot(self.data[i,0],self.data[i,1],'C'+str(j)+'.')
		plt.show()
		plt.plot(self.bestLogLikelihood[1:])
		plt.xlabel('Number of iterations')
		plt.ylabel('Log-likelihood')
		plt.show()

if __name__ == '__main__':
	a = EM()
	a.read_arguments()
	
	if type(a.numberOfClusters) == int:
		a.setClusterProbabilities()
		a.readData()
		start = time.time()
		a.findBestCluster()
		print ("Time taken is "+str(time.time()-start)+" s")
		a.visualize()
	
	else:
		i = 1
		x = [0]
		a.numberOfClusters = 1
		while True:
			a.setClusterProbabilities()
			a.readData()
			start = time.time()
			a.findBestCluster()
			prev = x[-1]
			print(a.logLikelihood[-1])
			x.append(a.BIC())
			if i>1 and x[-1]<=prev:
				a.numberOfClusters = i-1
				break
			else:
				a.numberOfClusters += 1 
				a.getBestParameters()
			i += 1	
		print ("Time taken is "+str(time.time()-start)+" s")
		print("The best number of clusters found using Bayesian Information Criterion is " + str(a.numberOfClusters))
		plt.plot(range(1,i+1),x[1:])
		plt.xlabel("Number of clusters")
		plt.ylabel("BIC")
		plt.show()
		a.BICvisualize()