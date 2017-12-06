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
timeCeiling = 60


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
	return (predictions == labels).sum() * 1.00 / labels.size

#####CAUTION AREA, STUDENT DEFINED METHODS!!!######
def eDist(xQ, xI, featureList):
	#euclidean distance function
	#assumes xQ and xI are lists with ranges given by positionRange
	#assumes tuples are of somewhat equal lenght (contain the range defined by positionRange)
	#positionRange should be a tuple with a start and stop index...
	#input: query tuple, dataset tuple, range of values to be considered
	totalSumOfSquares = 0.00
	try:
		for feature in featureList:			
			totalSumOfSquares += abs(xQ[feature] - xI[feature])**2
		dist = math.sqrt(totalSumOfSquares)
		return dist
	except:
		print('Error, no Euclidean distance returned...values passed xQ, xI were of length {}, {}.'.format(len(xQ), len(xI)))

def hDist(xQ, xI):
	#hamming distance function
	#assumes xQ and xI are lists with ranges given by positionRange
	#assumes tuples are of somewhat equal lenght (contain the range defined by positionRange)
	#positionRange should be a tuple with a start and stop index...
	d = 0
	booleanFeatures = ['can_off_P', 'can_off_S', 'can_off_H', 'can_inc_cha_ope_sea_INCUMBENT', 'can_inc_cha_ope_sea_CHALLENGER',\
		'can_inc_cha_ope_sea_OPEN']
	for feature in booleanFeatures:
		if xQ[feature] is not xI[feature]:
			d += 1
	return d

def stdDev(dataset, feature):
	average = dataset[feature].sum() / len(dataset[feature])
	sumD = 0
	for i in range(0, len(dataset[feature])):
		sumD += (i - average)**2
	return math.sqrt(sumD / (len(dataset) - 1))
	


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
		self.vote = {}

	def majorityVote(self):
		#k must be odd
		#print('self.vote is of length {}'.format(len(self.vote)))
		for canID in self.nearestNeighbors.keys():
			score = 0
			#print('testObj is {} and of type({})'.format(testObj, type(testObj)))
			#print('size of (self.nearestNeighbors.keys() is {}'.format(len(self.nearestNeighbors.keys())))
			for neighbor in self.nearestNeighbors[canID].iterkeys():
				score += self.nearestNeighbors[canID][neighbor]
			if self.k/2 < score:
				self.vote[canID] = True
			else:
				self.vote[canID] = False
			#print('score for ID {} is {} and the vote is {}.'.format(canID, score, self.vote[canID]))
			
	def train(self, features, ratio):
		#training logic here
		#input is list/array of features and labels
		self.trainingData, self.testingData = trainingTestData(self.dataset, ratio)

	def predict(self, features, labels):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		###STUDENT ENTRIES
		print('features in kNN predict is {}'.format(features))
		#sizeOfTrainingData = len(self.trainingData)
		#for testingIndex in range(0, len(self.testingData)):
		stopTime = timeCeiling + time.time()
		while time.time() < stopTime:
			for testingIndex in range(0, len(self.testingData)):
				#testingIndex = random.randint(0, len(self.testingData) - 1)
				testingID = self.testingData.iloc[testingIndex]['can_id']
				if testingID in self.nearestNeighbors.keys():
					currentMax = max(k for k in self.nearestNeighbors[testingID].iterkeys())
				else:	
					self.nearestNeighbors[testingID] = {}
					currentMax = None
				#for trainingIndex in range(0, len(self.trainingData)):		
				timeSlice = timeCeiling/len(self.testingData)
				sliceStopTime = time.time() + timeSlice
				while len(self.nearestNeighbors[testingID]) < self.k or time.time() < sliceStopTime:
					trainingIndex = random.randint(0, len(self.trainingData) - 1)
					label = bool(self.trainingData.loc[trainingIndex, 'winner'])
					regressionValues = features[:3]
					currentNeighborEDist = eDist(self.testingData.iloc[testingIndex], self.trainingData.iloc[trainingIndex], regressionValues)
					#currentNeighborHDist = hDist(self.testingData.iloc[testingIndex], self.trainingData.iloc[trainingIndex])
					#currentNeighborHDist = currentNeighborHDist / 2.00
					totalCurrentNeighborDist = currentNeighborEDist #+ currentNeighborHDist
					#attempt at normalizing the hamming distance, since types are mutually exclusive,
					#the max distance for Hamming is 2
					#and the boolean 'hot encoded' values are only 2
					#but does it make sense to add hammming dist to euclidean dist?
					if currentMax is None:
						self.nearestNeighbors[testingID][totalCurrentNeighborDist] = label
					else:
						currentMax = max(k for k in self.nearestNeighbors[testingID].iterkeys())				
						if (totalCurrentNeighborDist < currentMax) and \
						not (totalCurrentNeighborDist in self.nearestNeighbors[testingID].iterkeys()):
							self.nearestNeighbors[testingID][totalCurrentNeighborDist] = label
						else:
							pass


class Perceptron:
	def __init__(self, dataset, features):
		#Perceptron state here
		self.dataset = dataset
		self.trainingData = None
		self.testingData = None
		#randomly initialize weights for the features
		#weights between 0 and 1
		self.weights = {}
		for feature in features:
			self.weights[feature] = random.randint(0, 100) / 100.00
		self.predictions = []
			

	def train(self, featuresList, featuresListHot, labels, ratio):
		#print initial weights for comparison
		print('start of training: weights are now {}'.format(self.weights))
		#training logic here
		self.trainingData, self.testingData = trainingTestData(self.dataset, ratio)
		numericFeaturesList = featuresList[:3]
		cOfficeHotList = featuresListHot[3:6]
		cIncChalHotList = featuresListHot[6:9]
		print('cOfficeHotList is {}. \ncIncChalHotList is {}.'.format(cOfficeHotList, cIncChalHotList))
		#input is list/array of features and labels
		endTime = time.time() + timeCeiling
		while time.time() < endTime:
			index = random.randint(0, len(self.trainingData) - 1)
			label = bool(self.trainingData.loc[index, 'winner'])
			for nfeature in numericFeaturesList:
				#create the hyperplane to crossect as True or False (1, 0)
				currentOut = self.weights[nfeature] * float(self.trainingData.loc[index, nfeature])
				#since all data is normalized, the currentOut should be between 0 and 1
				#w = w_original + (desired_output - current_ouput) * input_value
				if currentOut < 0 and label is True:
					self.weights[nfeature] = self.weights[nfeature] + (0 - currentOut) * float(self.trainingData.loc[index, nfeature])
				elif currentOut > 0 and label is False:
					self.weights[nfeature] = self.weights[nfeature] + (1 - currentOut) * float(self.trainingData.loc[index, nfeature])
				else: 
					pass
			#'can_off', 'can_inc_cha_ope_sea'
			normcOfficeOut = self.weights['can_off'] * 1.00 
			if normcOfficeOut > 0 and label is False:
				self.weights['can_off'] = self.weights['can_off'] + (1.00 - normcOfficeOut) * normcOfficeOut
			elif normcOfficeOut < 0 and label is True:
				self.weights['can_off'] = self.weights['can_off'] + (0.00 - normcOfficeOut) * normcOfficeOut
			else:
				pass
			ncIncChalOut = self.weights['can_inc_cha_ope_sea'] * 1.00
			if ncIncChalOut > 0 and label is False:
				self.weights['can_inc_cha_ope_sea'] = self.weights['can_inc_cha_ope_sea'] + (1.00 - ncIncChalOut) * ncIncChalOut
			elif normcOfficeOut < 0 and label is True:
				self.weights['can_inc_cha_ope_sea'] = self.weights['can_inc_cha_ope_sea'] + (0.00 - ncIncChalOut) * ncIncChalOut
			else:
				pass

		print('finished Perceptron training...')
		print('weights are now {}'.format(self.weights))


	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		self.predictions = [(None, -1) for n in range(0, len(self.testingData))]
		for index in range(0, len(self.testingData)):
			results = []
			for feature in features:
				print('self.weights[feature] is {}'.format(self.weights[feature]))
				print('self.testingData[index] is of type {}'.format(type(self.testingData[index])))
				results.append(self.weights[feature] * self.testingData[index][feature])
			if max(results)	> 0:
				self.predictions[index] = (testingData[index], True)
			elif max(results) < 0:
				self.predictions[index] = (testingData[index], False)
			else:
				pass


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


#kNN
kNNallFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
kNNnDataSet = normalizeData(dataset, kNNallFeatures[:3])
kNNencodedDataSet = encodeData(kNNnDataSet, kNNallFeatures[3:])
features, labels = getNumpy(kNNencodedDataSet)
kNNObj = KNN(kNNencodedDataSet, 7)
ratio = 0.5
kNNObj.train(features, ratio)
kNNObj.predict(kNNallFeatures, labels)
kNNObj.majorityVote()
#match can_ids between predictions (kNNObj.vote['can_id']) and
#testing data (kNNObj.testingData['winner'])
predictions = []
actuals = []
for index in range(0, len(kNNObj.testingData)):
	canID = kNNObj.testingData.iloc[index]['can_id']
	predictions.append(kNNObj.vote[canID])
	actuals.append(kNNObj.testingData.iloc[index]['winner'])
evaluationResult = evaluate(predictions, actuals)
print('evaluation result is {}'.format(evaluationResult))


"""
An object is classified by a majority vote of its neighbors, with the object being assigned to 
the class most common among its k nearest neighbors (k is a positive integer, typically small). 
If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
"""
#Perceptron
PallFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
PallFeaturesHot = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off_P', 'can_off_S', 'can_off_H',\
 'can_inc_cha_ope_sea_INCUMBENT', 'can_inc_cha_ope_sea_CHALLENGER', 'can_inc_cha_ope_sea_OPEN']
PallFeaturesBools = [ 'can_off', 'can_inc_cha_ope_sea' ]
PnDataSet = normalizeData(dataset, PallFeaturesHot[:3])
PEncodedDataSet = encodeData(PnDataSet, PallFeaturesBools)
pfeatures, plabels = getNumpy(PEncodedDataSet)
ratio = 0.5
firstP = Perceptron(PEncodedDataSet, PallFeatures)
print('firstP has weights of {}'.format(firstP.weights))
#firstP.train(PallFeatures, PallFeaturesHot, plabels, ratio)
#firstP.predict(PallFeaturesHot)

#####NOTES SECTION#####
#dont forget the bias
#make sure to NORMALIZE the data