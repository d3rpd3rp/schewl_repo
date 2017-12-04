import pandas as pd 
import numpy as np
####ADDITIONAL LIBRARIES USED####
import math
import time
import random

#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")


global timeCeiling
timeCeiling = 20


#========================================== Data Helper Functions ==========================================

#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData

#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
	return pd.get_dummies(dataset, columns=categories)

#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
	tr = int(len(dataset)*ratio)
	return dataset[:tr], dataset[tr:]

#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
	features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
	labels = dataset["winner"].astype(int).values
	return features, labels

#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
	f, l = getNumpy(dataset)
	return f.tolist(), l.tolist()

#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
	predictions = np.array(solutions)
	labels = np.array(real)
	return (predictions == labels).sum() / labels.size

#####CAUTION AREA, STUDENT DEFINED METHODS!!!######
def eDist(xQ, xI, positionRange):
	#euclidean distance function
	#assumes xQ and xI are lists with ranges given by positionRange
	#assumes tuples are of somewhat equal lenght (contain the range defined by positionRange)
	#positionRange should be a tuple with a start and stop index...
	#input: query tuple, dataset tuple, range of values to be considered
	totalSumOfSquares = 0.00
	try:
		print('min(positionRange), max(positionRange) are {} and {}'.format(min(positionRange), max(positionRange)))
		#print('xQ and xI are {} and {}'.format(xQ, xI))
		for index in range(min(positionRange), max(positionRange)):
			totalSumOfSquares += abs(xQ[index] - xI[index])**2
		dist = math.sqrt(totalSumOfSquares)
		return dist
	except:
		print('Error, no Euclidean distance returned...values passed xQ, xI were of length {}, {}.'.format(len(xQ), len(xI)))

def hDist(xQ, xI, positionRange):
	#hamming distance function
	#assumes xQ and xI are lists with ranges given by positionRange
	#assumes tuples are of somewhat equal lenght (contain the range defined by positionRange)
	#positionRange should be a tuple with a start and stop index...
	try:
		print('min(positionRange), max(positionRange) are {} and {}'.format(min(positionRange), max(positionRange)))
		d = 0
		for index in range(min(positionRange), max(positionRange)):
			if xQ[index] is not xI[index]:
				d += 1
		return d
	except:
		print('Error, no Hamming distance returned...values passed xQ, xI were of length {}, {}.'.format(len(xQ), len(xI)))



#===========================================================================================================

class KNN:
	def __init__(self, dataset, k):
		#KNN state here
		#Feel free to add methods
		self.dataset = dataset
		self.k = k
		self.trainingData = None
		self.testingData = None
		self.nearestNeighbors = {}


	def train(self, features, ratio):
		#training logic here
		#input is list/array of features and labels
		self.trainingData, self.testingData = trainingTestData(self.dataset, ratio)
		#create dummy tuple for initialization of the nearest neighbors structure
		initList = []
		for j in range(0, len(features)):
			initList.append(2)
		for i in range(2, (self.k + 2)):
			self.nearestNeighbors[i] = [initList, i%2]

	def predict(self, features, labels):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		###STUDENT ENTRIES
		#will use the sum of all features to get started...
		#nearestNeighbors will be a tuple of the instance data, totalCurrentNeighborDist, and outcome
		#outcome is available from the training data
		
		ePositionRange = (0, 1, 2)
		hPositionRange = (3, 4)
		endTime = time.time() + timeCeiling
		#for testInstance in self.testingData:
		while time.time() < endTime:
			print('testInstance is {}'.format(testInstance))
			for trainInstance in self.trainingData:
				currentNeighborEDist = eDist(testInstance, trainInstance, ePositionRange)
				currentNeighborHDist = hDist(testInstance, trainInstance, hPositionRange)
				#attempt at normalizing the hamming distance, range is 0 through 6
				currentNeighborHDist = currentNeighborHDist / 6
				#adding another factor to represent that the numeric values are 3
				#and the boolean 'hot encoded' values are only 2
				totalCurrentNeighborDist = 2/5 * currentNeighborHDist + 3/5 * currentNeighborEDist
				print('self nN is: ')
				for k, v in self.nearestNeighbors.iteritems():
					print('k, v are {} , {}'.format(k, v))
				currentMax = max(k for k, v in self.nearestNeighbors.iteritems())
				print('currentMax is {} before comparison with totalDist...'.format(currentMax))
				print('trainInstance is {}'.format(trainInstance))
				if totalCurrentNeighborDist < currentMax or currentMax is None:
					print('nearestNeighbors[currentMax] is {} before comparison with totalDist...'.format(self.nearestNeighbors[currentMax]))
					del self.nearestNeighbors[currentMax]
					self.nearestNeighbors[totalCurrentNeighborDist] = trainInstance
					print('size of the dictionary for NN is now {}.'.format(len(self.nearestNeighbors)))


class Perceptron:
	def __init__(self, dataset, features):
		#Perceptron state here
		self.dataset = dataset
		self.trainingData, self.testingData = trainingTestData(self.dataset, ratio)
		self.nearestNeighbors = {}
		#randomly initialize weights for the features
		#weights between 0 and 1
		self.weights = {}
		for feature in features:
			self.weights[feature] = random.randint(0, 100) / 100.00
		#print('initial weights are: ')
		#print(self.weights)
			

	def train(self, featuresList, featuresListHot, labels, ratio):
		#print initial weights for comparison
		print('start of training: weights are now {}'.format(self.weights))
		#training logic here
		self.trainingData, self.testingData = trainingTestData(self.dataset, ratio)
		numericFeaturesList = featuresList[:3]
		cOfficeHotList = featuresListHot[3:6]
		cIncChalHotList = featuresListHot[6:9]
		print('cOfficeHotList is {}. \n cIncChalHotList is {}.'.format(cOfficeHotList, cIncChalHotList))
		#input is list/array of features and labels
		ePositionRange = (0, 1, 2)
		hPositionRange = (3, 4)
		endTime = time.time() + timeCeiling
		while time.time() < endTime:
			index = random.randint(0, len(self.trainingData) - 1)
			label = bool(self.trainingData.loc[index, 'winner'])
			for nfeature in numericFeaturesList:
				#create the hyperplane to crossect as True or False (1, 0)
				currentOut = self.weights[nfeature] * float(self.trainingData.loc[index, nfeature])
				#since all data is normalized, the currentOut should be between 0 and 1
				#w = w_original + (desired_output - current_ouput) * input_value
				if currentOut < 0.50 and label is True:
					self.weights[nfeature] = self.weights[nfeature] + (0 - currentOut) * float(self.trainingData.loc[index, nfeature])
					#print('corrected weight, currentOut is {} and label is {}'.format(currentOut, label))
				elif currentOut > 0.50 and label is False:
					self.weights[nfeature] = self.weights[nfeature] + (0 - currentOut) * float(self.trainingData.loc[index, nfeature])
					#print('corrected weight, currentOut is {} and label is {}'.format(currentOut, label))
				else: 
					pass
				#'can_off', 'can_inc_cha_ope_sea'
			cOfficeOut = 0
			for cOfficeFeature in cOfficeHotList:
				#print('current object...bfeature is {} and trainingData bfeature is {}'.format(bfeature, self.trainingData.loc[index, bfeature]))
				cOfficeOut += self.trainingData.loc[index, cOfficeFeature]
			print('total sum from cOfficeHotList is {}'.format(cOfficeOut))
			normcOfficeOut = self.weights['can_off'] * cOfficeOut / 3.00 
			if normcOfficeOut > 0.50 and label is False:
				self.weights['can_off'] = self.weights['can_off'] + (0 - normcOfficeOut) * normcOfficeOut
			elif normcOfficeOut < 0.50 and label is True:
				self.weights['can_off'] = self.weights['can_off'] + (1 - normcOfficeOut) * normcOfficeOut
			else:
				pass
			cIncChalHotOut = 0
			for cIncChalFeature in cIncChalHotList:
				cIncChalHotOut += self.trainingData.loc[index, cIncChalFeature]
			print('total sum from cOfficeHotList is {}'.format(cIncChalHotOut))
			ncIncChalOut = self.weights['can_inc_cha_ope_sea'] * cIncChalHotOut / 3.00
			if ncIncChalOut > 0.50 and label is False:
				self.weights['can_inc_cha_ope_sea'] = self.weights['can_inc_cha_ope_sea'] + (0 - ncIncChalOut) * ncIncChalOut
			elif normcOfficeOut < 0.50 and label is True:
				self.weights['can_inc_cha_ope_sea'] = self.weights['can_inc_cha_ope_sea'] + (1 - ncIncChalOut) * ncIncChalOut
			else:
				pass
 

			



				#print('currentOut is now {}'.format(currentOut))

		print('finished Perceptron training...')
		print('weights are now {}'.format(self.weights))


	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		features = None

class MLP:
	def __init__(self):
		#Multilayer perceptron state here
		#Feel free to add methods
		states = None

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		features = None

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		features = None

class ID3:
	def __init__(self):
		#Decision tree state here
		#Feel free to add methods
		states = None

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		features = None

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		features = None

"""
print('testing output formats and functions...')
#all_categories = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
categories = ['can_off', 'can_inc_cha_ope_sea']
print('firstly, hot encode the panda data set...')
encodedDataSet = encodeData(dataset, categories)
print('getPythonList(encodedDataSet) returns type {}'.format(getPythonList(encodedDataSet)))
#when do we need to normalize? what is normalize?
"""




#kNN
kNNallFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
kNNnDataSet = normalizeData(dataset, kNNallFeatures[:3])
kNNencodedDataSet = encodeData(kNNnDataSet, kNNallFeatures[3:])
features, labels = getNumpy(kNNencodedDataSet)
#print('kNN numpy looks like this: \n{}'.format(features))
fiveNN = KNN(kNNencodedDataSet, 5)
ratio = 0.5
#commenting out for testing...03 December 2017 12:39
#fiveNN.train(kNNallFeatures, ratio)
#fiveNN.predict(allFeatures, labels)
#print('nearest neighbors for fiveNN object after predict is {}'.format(fiveNN.nearestNeighbors))



"""
print('printing out first class object creation for KNN...')
print('fiveNN.fullEncodedDataSetList is:\n{}'.format(fiveNN.dataset))
print('attempting to find the feature / label portions of the list...')
print('fiveNN.fullEncodedDataSetList is of length {}'.format(len(fiveNN.dataset)))
print('methinks labels are in pos 0 which are of length {}'.format(len(fiveNN.dataset[0])))
print('and then in equal size the pos 1 which is of length {}.'.format(len(fiveNN.dataset[1])))
"""



#tmpArray = [[no, netcon, totloa, canoff, canInc] for no, netcon, totloa, canoff, canInc in enumerate(fiveNN.trainingData)]
#print('tmpArray:\n {}.'.format(tmpArray))



"""
An object is classified by a majority vote of its neighbors, with the object being assigned to 
the class most common among its k nearest neighbors (k is a positive integer, typically small). 
If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
"""
PallFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
PallFeaturesHot = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off_P', 'can_off_S', 'can_off_H',\
 'can_inc_cha_ope_sea_INCUMBENT', 'can_inc_cha_ope_sea_CHALLENGER', 'can_inc_cha_ope_sea_OPEN']
PallFeaturesBools = [ 'can_off', 'can_inc_cha_ope_sea' ]
PnDataSet = normalizeData(dataset, PallFeaturesHot[:3])
#print('PallFeatures[:3] is {}'.format(PallFeatures[:3]))
PEncodedDataSet = encodeData(PnDataSet, PallFeaturesBools)
pfeatures, plabels = getNumpy(PEncodedDataSet)
#print('pfeatures looks like this: \n{}'.format(pfeatures))
ratio = 0.5
firstP = Perceptron(PEncodedDataSet, PallFeatures)
print('firstP has weights of {}'.format(firstP.weights))
firstP.train(PallFeatures, PallFeaturesHot, plabels, ratio)

#####NOTES SECTION#####
#dont forget the bias
#make sure to NORMALIZE the data