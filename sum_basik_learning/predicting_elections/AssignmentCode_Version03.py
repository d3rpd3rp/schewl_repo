import pandas as pd 
import numpy as np
import math, random
import time

#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")

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
	return (predictions == labels).sum() / float(labels.size)


#===========================================================================================================

class KNN:
	def __init__(self):
		#KNN state here
		#Feel free to add methods
		self.dataset = dataset
		self.k = 7
		self.nearestNeighbors = {}
		self.vote = {}
		self.ratio = 0.10
		#establishment, definition of datasets
		kNNallFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
		kNNnDataSet = normalizeData(dataset, kNNallFeatures[:3])
		kNNencodedDataSet = encodeData(kNNnDataSet, kNNallFeatures[3:])
		self.trainingData, self.testingData = trainingTestData(kNNencodedDataSet, self.ratio)
		
	#####CAUTION AREA, STUDENT DEFINED METHODS!!!######
	def eDist(self, xQ, xI, featureList):
		#euclidean distance function
		#assumes xQ and xI are lists with ranges given by positionRange
		#assumes tuples are of somewhat equal lenght (contain the range defined by positionRange)
		#positionRange should be a tuple with a start and stop index...
		#input: query tuple, dataset tuple, range of values to be considered
		totalSumOfSquares = 0.00
		for feature in featureList:			
			totalSumOfSquares += abs(xQ[feature] - xI[feature])**2
		dist = math.sqrt(totalSumOfSquares)
		return dist


	def hDist(self, xQ, xI):
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

	def majorityVote(self):
		for canID in self.nearestNeighbors.keys():
			score = 0
			for neighbor in self.nearestNeighbors[canID].iterkeys():
				score += self.nearestNeighbors[canID][neighbor]
			if self.k/2 < score:
				self.vote[canID] = True
			else:
				self.vote[canID] = False
	#########################################
	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels
		print('this is a lazy learning model, nothing should happen here...')


	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		startTime = time.time()
		for testingIndex in range(0, len(self.testingData)):
			testingID = self.testingData.iloc[testingIndex]['can_id']
			self.nearestNeighbors[testingID] = {}
			currentMax = None
			for trainingIndex in range(0, len(self.trainingData)):		
				trainingLabel = bool(self.trainingData.loc[trainingIndex, 'winner'])
				continuousValues = features[:3]
				currentNeighborEDist = self.eDist(self.testingData.iloc[testingIndex], self.trainingData.iloc[trainingIndex], continuousValues)
				currentNeighborHDist = self.hDist(self.testingData.iloc[testingIndex], self.trainingData.iloc[trainingIndex])
				#an attempt to normalize the Hamming Distance, max distance of 2.00 (both attributes differ)
				currentNeighborHDist = currentNeighborHDist / 2.00
				totalCurrentNeighborDist = currentNeighborEDist + currentNeighborHDist
				#and the boolean 'hot encoded' values are only 2
				#but does it make sense to add hammming dist to euclidean dist?
				if currentMax is None or len(self.nearestNeighbors()) < self.k:
					self.nearestNeighbors[testingID][totalCurrentNeighborDist] = trainingLabel
				else:
					currentMax = max(k for k in self.nearestNeighbors[testingID].keys())				
					if (totalCurrentNeighborDist < currentMax):
						self.nearestNeighbors[testingID][totalCurrentNeighborDist] = trainingLabel
					else:
						pass
		endTime = time.time()
		print('the prediction took {} s.'.format(abs(startTime - endTime)))
		self.majorityVote()
		predictions = []
		for ID in self.testingData['can_id']:
			predictions.append(self.vote[ID])
		return predictions


class Perceptron:
	def __init__(self):
		#Perceptron state here
		self.rawFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
		self.encodedFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off_P', 'can_off_S', 'can_off_H',\
 		'can_inc_cha_ope_sea_INCUMBENT', 'can_inc_cha_ope_sea_CHALLENGER', 'can_inc_cha_ope_sea_OPEN']
		#
		self.trainingData = None
		#numpy objects
		self.trainingFeatures = None
		self.trainingLabels = None
		#
		self.testingData = None
		#numpy objects
		self.testingFeatures = None
		self.testingLabels = None
		self.ratio = 0.5
		#randomly initialize weights for the features
		#weights between -0.01 and 0.01
		self.weights = {}
		self.predictions = []
		self.bias = 1.000000

	def preprocess(self, dataset):
		nPDataset = normalizeData(dataset, self.rawFeatures[:3])
		encNPDataset = encodeData(nPDataset, self.rawFeatures[3:])
		self.trainingData, self.testingData = trainingTestData(encNPDataset, self.ratio)
		self.trainingFeatures, self.trainingLabels = getNumpy(self.trainingData)
		self.testingFeatures, self.testingLabels = getNumpy(self.trainingData)
		#initialize weights, for continuous labels
		#ASSUMPTION, the first three labels are continuous, 
		#the second two (represented by 6 fields) are discrete
		for i in range(0, len(self.encodedFeatures)):
			scalar = random.random() / 100000.000000
			print('scalar is {}'.format(scalar))
			self.weights[i] = random.choice([0.000000 - scalar, scalar])

	def train(self):
		#training logic here
		#input is list/array of features and labels
		#print initial weights for comparison
		print('start of training: weights are now {}'.format(self.weights))
		#training logic here		
		#input is list/array of features and labels
		continuousFeatures = self.rawFeatures[:3]
		booleanFeatures = self.encodedFeatures[3:]
		adjustments = [0, 0, 0, 0, 0, 0, 0, 0, 0]
		for trainingInstance, trainingLabel in zip(self.trainingFeatures, self.trainingLabels):
			#create the hyperplane to crossect as True or False (1, 0)
			#since all data is normalized, the currentOut should be between 0 and 1
			#w = w_original + (desired_output - current_ouput) * input_value
			for cfposition in range(0, len(self.encodedFeatures)):
				currentOut = self.weights[cfposition] * trainingInstance[cfposition] + self.bias
				if currentOut < 0 and trainingLabel == 1:
					self.weights[cfposition] = self.weights[cfposition] + (1.000000 - currentOut) * trainingInstance[cfposition]
					self.bias = self.weights[cfposition] * trainingInstance[cfposition] + self.bias
					adjustments[cfposition] += 1 
				elif currentOut > 0 and trainingLabel == 0:
					self.weights[cfposition] = self.weights[cfposition] + (0.000000 - currentOut) * trainingInstance[cfposition]
					self.bias = self.weights[cfposition] * trainingInstance[cfposition] - self.bias
					adjustments[cfposition] += 1 
				else:
					pass

		print('end of training: bias is {}, weights are now {}'.format(self.bias, self.weights))
		print('{} adjustments were made.'.format(adjustments))

	def predict(self):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
		for testingInstance in self.testingFeatures:
			#find largest vector
			largestScalar = 0
			lposition = None
			for fposition in range(0, len(self.weights)):
				#print('self.weights[{}], testingInstance, bias are {}, {}, {}'.format(fposition, self.weights[fposition], testingInstance[fposition],  self.bias))
				scalar = abs(self.weights[fposition] * testingInstance[fposition] + self.bias) 
				#print('scalar value is {}'.format(scalar))
				if largestScalar < scalar:
					 largestScalar = scalar
					 lposition = fposition
				else:
					pass
			if 0 < self.weights[lposition] * testingInstance[lposition]:
				self.predictions.append(1)
			elif self.weights[lposition] * testingInstance[lposition] < 0:
				self.predictions.append(0)
			else:
				print('weights[{}] * testingInstance[{}] is {}'.format(lposition, lposition, (self.weights[lposition] * testingInstance[lposition])))
		incorrectCount = 0
		for prediction, label in zip(self.predictions, self.testingLabels):
			if prediction != label:
				incorrectCount += 1
		print('{} incorrect, sizes of predictions and testingLabels are {} and {}'.format(incorrectCount, len(self.predictions), len(self.testingLabels)))



class MLP:
	def __init__(self):
		#Multilayer perceptron state here
		#Feel free to add methods

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features

"""
class ID3:
	def __init__(self):
		#Decision tree state here
		#Feel free to add methods

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features
"""
