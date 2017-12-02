import pandas as pd 
import numpy as np

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
	return (predictions == labels).sum() / labels.size

#===========================================================================================================

class KNN:
	def __init__(self):
		#KNN state here
		#Feel free to add methods

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features

class Perceptron:
	def __init__(self):
		#Perceptron state here
		#Feel free to add methods

	def train(self, features, labels):
		#training logic here
		#input is list/array of features and labels

	def predict(self, features):
		#Run model here
		#Return list/array of predictions where there is one prediction for each set of features

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
