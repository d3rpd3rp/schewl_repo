import pandas as pd
import numpy as np
import math, random
import time

# Data with features and target values
# Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
# Helper functions are provided so you shouldn't need to learn Pandas
dataset = pd.read_csv("data.csv")


# ========================================== Data Helper Functions ==========================================

# Normalize values between 0 and 1
# dataset: Pandas dataframe
# categories: list of columns to normalize, e.g. ["column A", "column C"]
# Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData


# Encode categorical values as mutliple columns (One Hot Encoding)
# dataset: Pandas dataframe
# categories: list of columns to encode, e.g. ["column A", "column C"]
# Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
    return pd.get_dummies(dataset, columns=categories)


# Split data between training and testing data
# dataset: Pandas dataframe
# ratio: number [0, 1] that determines percentage of data used for training
# Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
    tr = int(len(dataset) * ratio)
    return dataset[:tr], dataset[tr:]


# Convenience function to extract Numpy data from dataset
# dataset: Pandas dataframe
# Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
    features = dataset.drop(["can_id", "can_nam", "winner"], axis=1).values
    labels = dataset["winner"].astype(int).values
    return features, labels


# Convenience function to extract data from dataset (if you prefer not to use Numpy)
# dataset: Pandas dataframe
# Return: features list and corresponding labels as a list
def getPythonList(dataset):
    f, l = getNumpy(dataset)
    return f.tolist(), l.tolist()


# Calculates accuracy of your models output.
# solutions: model predictions as a list or numpy array
# real: model labels as a list or numpy array
# Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
    predictions = np.array(solutions)
    labels = np.array(real)
    return (predictions == labels).sum() / float(labels.size)


# ===========================================================================================================

class KNN:
    def __init__(self):
        # KNN state here
        # Feel free to add methods
        self.k = 5

    #####CAUTION AREA, STUDENT DEFINED METHODS!!!######
    def eDist(self, xQ, xI):
        # euclidean distance function
        # assumes xQ and xI are lists with ranges given by positionRange
        totalSumOfSquares = 0.00
        for i in range(0, 3):
            totalSumOfSquares += abs(xQ[i] - xI[i]) ** 2
        dist = math.sqrt(totalSumOfSquares)
        return dist

    def hDist(self, xQ, xI):
        # hamming distance function
        d = 0
        for i in range(3, 9):
            if xQ[i] is not xI[i]:
                d += 1
        return d


    #########################################
    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        print('this is a lazy learning model, nothing should happen here...')
        self.trainingFeatures = features
        self.trainingLabels = labels

    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        self.vote = [None] * len(features)
        self.predictions= []
        startTime = time.time()
        for testingIndex in range(0, len(features)):
            #normalized Euclidean values are between [0,1], max hamming distance 2, so setting the value to 5
            nearestNeighbors = [(5, None)] * self.k
            for trainingIndex in range(0, len(self.trainingFeatures)):
                trainingLabel = bool(self.trainingLabels[trainingIndex])
                currentNeighborEDist = self.eDist(features[testingIndex], self.trainingFeatures[trainingIndex])
                currentNeighborHDist = self.hDist(features[testingIndex], self.trainingFeatures[trainingIndex])
                # an attempt to normalize the Hamming Distance, max distance of 2.00 (both attributes differ)
                currentNeighborHDist = currentNeighborHDist / 2.00
                totalCurrentNeighborDist = currentNeighborEDist + currentNeighborHDist
                # and the boolean 'hot encoded' values are only {0, 1, 2}
                # but does it make sense to add hammming dist to euclidean dist?
                currentMaxNeighbor = max(nearestNeighbors, key=lambda x:x[0])
                print('currentMaxNeighbor[0] is {}, totalCurrentNeighborDist is {}'.format(currentMaxNeighbor[0], totalCurrentNeighborDist))
                print('nearestNeighbors is {}'.format(nearestNeighbors))
                if totalCurrentNeighborDist < currentMaxNeighbor[0]:
                    for index, pair in enumerate(nearestNeighbors):
                        print('pair[0] is {} and currentMaxNeighbor is {}'.format(pair[0], currentMaxNeighbor))
                        if pair[0] == currentMaxNeighbor[0]:
                            nearestNeighbors[index] = (totalCurrentNeighborDist, trainingLabel)
                            break
            score = 0
            print('nearestNeightbors is {}'.format(nearestNeighbors))
            print('trainingLabel is {}'.format(trainingLabel))
            for pair in nearestNeighbors:
                print('pair is {}'.format(pair))
                score += int(pair[1])
            print('score is {}'.format(score))
            if (self.k / 2) <= score:
                self.vote[testingIndex] = True
            else:
                self.vote[testingIndex] = False
            print('vote is {}'.format(self.vote[testingIndex]))
        endTime = time.time()
        print('the prediction took {} s.'.format(abs(startTime - endTime)))

        return (self.vote)


class Perceptron:
    def __init__(self):
        # Perceptron state here
        # randomly initialize weights for the features
        # weights between -0.01 and 0.01
        self.weights = []
        self.bias = 1.000000

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        # print initial weights for comparison
        # initialize weights, for continuous labels
        # ASSUMPTION, the first three labels are continuous,
        # the second two (represented by 6 fields) are discrete
        for i in range(0, features.shape[1]):
            scalar = random.random() / 100000.000000
            print('scalar is {}'.format(scalar))
            self.weights.append(random.choice([0.000000 - scalar, scalar]))
        print('start of training: weights are now {}'.format(self.weights))
        # training logic here
        adjustments = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        learningRate = 0.1000000
        for trainingInstance, trainingLabel in zip(features, labels):
            # create the hyperplane to crossect as True or False (1, 0)
            # since all data is normalized, the currentOut should be between 0 and 1
            # w = w_original + (desired_output - current_ouput) * input_value
            for cfposition in range(0, features.shape[1]):
                currentOut = self.weights[cfposition] * trainingInstance[cfposition] + self.bias
                if currentOut < 0 and trainingLabel == 1:
                    self.weights[cfposition] = self.weights[cfposition] + learningRate * (1.000000 - currentOut) * \
                                                                          trainingInstance[cfposition]
                    self.bias = self.weights[cfposition] * trainingInstance[cfposition] + self.bias
                    adjustments[cfposition] += 1
                elif 0 <= currentOut and trainingLabel == 0:
                    self.weights[cfposition] = self.weights[cfposition] + learningRate * (0.000000 - currentOut) * \
                                                                          trainingInstance[cfposition]
                    self.bias = self.weights[cfposition] * trainingInstance[cfposition] - self.bias
                    adjustments[cfposition] += 1
                else:
                    pass

        print('end of training: bias is {}, weights are now {} and sized {}'.format(self.bias, self.weights,
                                                                                    len(self.weights)))
        print('{} adjustments were made.'.format(adjustments))

    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        predictions = []
        for testingInstance in features:
            print('testingInstance is {}'.format(testingInstance))
            # find largest vector
            largestScalar = 0
            lposition = None
            for fposition in range(0, len(self.weights)):
                scalar = abs(self.weights[fposition] * testingInstance[fposition])
                if largestScalar < scalar:
                    largestScalar = scalar
                    lposition = fposition
                else:
                    pass
            print('lposition is {} and largestScalar is {}'.format(lposition, largestScalar))
            print('testingInstance[{}] is {}'.format(lposition, testingInstance[lposition]))
            if 0 <= self.weights[lposition] * testingInstance[lposition] + self.bias:
                predictions.append(1)
            elif self.weights[lposition] * testingInstance[lposition] + self.bias < 0:
                predictions.append(0)
            else:
                print('weights[{}] * testingInstance[{}] is {}'.format(lposition, lposition, (
                    self.weights[lposition] * testingInstance[lposition])))

        return (predictions)


class MLP:
    def __init__(self):
        # Multilayer perceptron state here
        # Feel free to add methods
        self.L0P0 = Perceptron()
        self.L0P1 = Perceptron()
        self.L0P2 = Perceptron()
        self.L1P0 = Perceptron()
        self.L1P1 = Perceptron()
        self.L2P0 = Perceptron()
        self.outputs = {}
        self.outputError = {}


    def activationFunction(self, value):
        output = 1.000000 / (1.000000 + math.exp(1) ** (value))
        return output

    def squaredError(self, outputP, label):
        lastError = self.outputError[outputP]
        currentError = 0.500000 * (self.outputError[outputP] - label) ** 2
        if abs(lastError - currentError) < 0.000001:
            return True
        else:
            return False

    def train(self, features, labels):
        # training logic here
        # input is list/array of features and labels
        for i in range(0, features.shape[1]):
            scalar = random.random() / 100000.000000
            self.L0P0.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000 )
            scalar = random.random() / 100000.000000
            self.L0P1.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000 )
            scalar = random.random() / 100000.000000
            self.L0P2.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000 )
            scalar = random.random() / 100000.000000
            self.L1P0.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
            scalar = random.random() / 100000.000000
            self.L1P1.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
            scalar = random.random() / 100000.000000
            self.L2P0.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
        learningRate = 0.100000
        stdError = 1.000000
        for trainingInstance, trainingLabel in zip(features, labels):
            # create the hyperplane to crossect as True or False (1, 0)
            # since all data is normalized, the currentOut should be between 0 and 1
            # weight = weight_prev - (learningRate * (expectedTarget - currentCalc) * currentCalc * (1 - currentCalc) * featureValue
            # currentCalc = weight * featureValue
            self.outputs[self.L0P0] = 0
            self.outputs[self.L0P1] = 0
            self.outputs[self.L0P2] = 0
            for L0fposition in range(0, features.shape[1]):
                self.L0P0.weights[L0fposition] = self.L0P0.weights[L0fposition] - (learningRate * stdError)
                self.L0P1.weights[L0fposition] = self.L0P0.weights[L0fposition] - (learningRate * stdError)
                self.L0P2.weights[L0fposition] = self.L0P0.weights[L0fposition] - (learningRate * stdError)
                self.L0P0.bias = self.L0P0.bias - (learningRate * stdError)
                self.L0P1.bias = self.L0P1.bias - (learningRate * stdError)
                self.L0P2.bias = self.L0P2.bias - (learningRate * stdError)
                self.outputs[self.L0P0] += self.L0P0.weights[L0fposition] * trainingInstance[L0fposition] + self.L0P0.bias
                """
                if self.outputs[self.L0P0] <= 0 and trainingLabel == 1:
                    self.L0P0.weights[cfposition] = self.L0P0.weights[cfposition] - (learningRate - self.outputs[self.L0P0]) * self.outputs[self.L0P0] * (1.000000 - self.outputs[self.L0P0]) * trainingInstance[cfposition]
                    adjustments[cfposition] += 1
                elif 0 < self.outputs[self.L0P0] and trainingLabel == 0:
                    self.L0P0.weights[cfposition] = self.L0P0.weights[cfposition] - (learningRate - self.outputs[self.L0P0]) * self.outputs[self.L0P0] * (1.000000 - self.outputs[self.L0P0]) * trainingInstance[cfposition]
                    adjustments[cfposition] += 1
                else:
                    pass
                """
                self.outputs[self.L0P1] += self.L0P0.weights[L0fposition] * trainingInstance[L0fposition] + self.L0P1.bias
                """
                if self.outputs[self.L0P1] <= 0 and trainingLabel == 1:
                    self.L0P1.weights[cfposition] = self.L0P1.weights[cfposition] - (learningRate - self.outputs[self.L0P1]) * self.outputs[self.L0P1] * (1.000000 - self.outputs[self.L0P1]) * trainingInstance[cfposition]
                adjustments[cfposition] += 1
                elif 0 < self.outputs[self.L0P1] and trainingLabel == 0:
                    self.L0P1.weights[cfposition] = self.L0P1.weights[cfposition] - (learningRate - self.outputs[self.L0P1]) * self.outputs[self.L0P0] * (1.000000 - self.outputs[self.L0P0]) * trainingInstance[cfposition]
                    adjustments[cfposition] += 1
                else:
                    pass
                """
                self.outputs[self.L0P2] += self.L0P2.weights[L0fposition] * trainingInstance[L0fposition] + self.L0P2.bias
                """
                if self.outputs[self.L0P2] <= 0 and trainingLabel == 1:
                    self.L0P2.weights[cfposition] = self.L0P2.weights[cfposition] - (learningRate - self.outputs[self.L0P2]) * self.outputs[self.L0P2] * (1.000000 - self.outputs[self.L0P2]) * trainingInstance[cfposition]
                elif 0 < self.outputs[self.L0P1] and trainingLabel == 0:
                    self.L0P1.weights[cfposition] = self.L0P1.weights[cfposition] - (learningRate - self.outputs[self.L0P1]) * self.outputs[self.L0P0] * (1.000000 - self.outputs[self.L0P0]) * trainingInstance[cfposition]
                    adjustments[cfposition] += 1
                else:
                    pass
                """
            self.outputs[self.L0P0] = self.activationFunction(self.outputs[self.L0P0])
            self.outputs[self.L0P1] = self.activationFunction(self.outputs[self.L0P1])
            self.outputs[self.L0P2] = self.activationFunction(self.outputs[self.L0P2])
            self.outputs[self.L1P0] = 0
            self.outputs[self.L1P1] = 0
            for L1fposition in range(0, features.shape[1]):
                self.L1P0.weights[L1fposition] = self.L1P0.weights[L1fposition] - (learningRate * stdError)
                self.L1P1.weights[L1fposition] = self.L1P1.weights[L1fposition] - (learningRate * stdError)
                for inputNode in [self.L0P0, self.L0P1, self.L0P2]:
                    self.outputs[self.L1P0] += self.L1P0.weights[L1fposition] * self.outputs[inputNode] + self.L1P0.bias
                    self.outputs[self.L1P1] += self.L1P1.weights[L1fposition] * self.outputs[inputNode] + self.L1P1.bias
            self.outputs[self.L1P0] = self.activationFunction(self.outputs[self.L1P0])
            self.outputs[self.L1P1] = self.activationFunction(self.outputs[self.L1P1])
            self.outputs[self.L2P0] = 0
            for L2position in range(0, features.shape[1]):
                self.L2P0.weights[L1fposition] = self.L2P0.weights[L1fposition] - (learningRate * stdError)
                for hiddenNode in [self.L1P0, self.L1P1]:
                    self.outputs[self.L2P0] += self.outputs[hiddenNode] + self.L2P0.bias
            self.outputs[self.L2P0] = self.activationFunction(self.outputs[self.L2P0])

            stdError = (self.outputs[self.L2P0] - trainingLabel)**2
            print('error is {}'.format(stdError))



    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        nothing = None


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

"""
phillip routine
kNN = KNN()
train_dataset, test_dataset = trainingTestData(dataset, train_ratio)
train_features, train_labels = preprocess(train_dataset)
kNN.train(train_features, train_labels)
test_features, test_labels = preprocess(test_dataset)
predictions = kNN.predict(test_features)
"""

# variables for all classes#
baseFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
hotEncFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off_P', 'can_off_S', 'can_off_H', \
                  'can_inc_cha_ope_sea_INCUMBENT', 'can_inc_cha_ope_sea_CHALLENGER', 'can_inc_cha_ope_sea_OPEN']
nDataSet = normalizeData(dataset, baseFeatures[:3])
encNormDataset = encodeData(nDataSet, baseFeatures[3:])

"""
# KNN
KNNRatio = 0.1
KNNtrainingData, KNNtestingData = trainingTestData(encNormDataset, KNNRatio)
KNNtrainingFeatures, KNNtrainingLabels = getNumpy(KNNtrainingData)
KNNtestingFeatures, KNNtestingLabels = getNumpy(KNNtestingData)

K = KNN()
K.train(KNNtrainingFeatures, KNNtrainingLabels)
predictions = K.predict(KNNtestingFeatures)

accuracy = evaluate(KNNtestingLabels, predictions)
print('accuracy for KNN is {}%'.format(accuracy))

"""
"""
#Perceptron
PRatio = 0.5
PtrainingData, PtestingData = trainingTestData(encNormDataset, PRatio)
PtrainingFeatures, PtrainingLabels = getNumpy(PtrainingData)
PtestingFeatures, PtestingLabels = getNumpy(PtestingData)
P = Perceptron()
#print('features for P are of type {} and numpy shape {}'.format(type(PtrainingFeatures), PtrainingFeatures.shape))
P.train(PtrainingFeatures, PtrainingLabels)
predictions = P.predict(PtestingFeatures)
print('size of predictions is {} and size of PtestingLabels is {}'.format(len(predictions), len(PtestingLabels)))
accuracy = evaluate(PtestingLabels, predictions)
print('accuracy for KNN is {}%'.format(accuracy))
"""


#MLP
MLPRatio = 0.5
MLPtrainingData, MLPtestingData = trainingTestData(encNormDataset, MLPRatio)
MLPtrainingFeatures, MLPtrainingLabels = getNumpy(MLPtrainingData)
MLPtestingFeatures, MLPtestingLabels = getNumpy(MLPtestingData)
M = MLP()
M.train(MLPtrainingFeatures, MLPtrainingLabels)

