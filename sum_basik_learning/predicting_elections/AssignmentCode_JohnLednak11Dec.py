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
        self.predictions = []
        startTime = time.time()
        for testingIndex in range(0, len(features)):
            # normalized Euclidean values are between [0,1], max hamming distance 2, so setting the value to 5
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
                currentMaxNeighbor = max(nearestNeighbors, key=lambda x: x[0])
                print('currentMaxNeighbor[0] is {}, totalCurrentNeighborDist is {}'.format(currentMaxNeighbor[0],
                                                                                           totalCurrentNeighborDist))
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
        self.outputSums = {}

    def activationFunction(self, list):
        value = np.sum(list)
        output = 1.000000 / (1.000000 + math.exp(1) ** (-value))
        return output

    def train(self, features, labels):
        # training logic here
        prevStdError = 0
        stdError = float('inf')
        # input is list/array of features and labels
        for p in [self.L0P0, self.L0P1, self.L0P2, self.L1P0, self.L1P1, self.L2P0]:
            self.outputs[p] = []
            self.outputSums[p] = 0
        errorCount = 0
        for i in range(0, features.shape[1]):
            scalar = random.random() / 100000.000000
            self.L0P0.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
            scalar = random.random() / 100000.000000
            self.L0P1.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
            scalar = random.random() / 100000.000000
            self.L0P2.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
            scalar = random.random() / 100000.000000
            self.L1P0.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
            scalar = random.random() / 100000.000000
            self.L1P1.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
            scalar = random.random() / 100000.000000
            self.L2P0.weights.append(random.choice([0.000000 - scalar, scalar]) + 0.100000)
        learningRate = 0.0100000
        for trainingInstance, trainingLabel in zip(features, labels):
            # create the hyperplane to crossect as True or False (1, 0)
            # since all data is normalized, the currentOut should be between 0 and 1
            # weight = weight_prev - (learningRate * (expectedTarget - currentCalc) * currentCalc * (1 - currentCalc) * featureValue
            # currentCalc = weight * featureValue
            for L0fposition in range(0, features.shape[1]):
                self.outputs[self.L0P0] = []
                self.outputs[self.L0P1] = []
                self.outputs[self.L0P2] = []
                self.outputs[self.L0P0].append(
                    self.L0P0.weights[L0fposition] * trainingInstance[L0fposition] + self.L0P0.bias)
                self.outputs[self.L0P1].append(
                    self.L0P0.weights[L0fposition] * trainingInstance[L0fposition] + self.L0P1.bias)
                self.outputs[self.L0P2].append(
                    self.L0P2.weights[L0fposition] * trainingInstance[L0fposition] + self.L0P2.bias)
            self.outputSums[self.L0P0] = self.activationFunction(self.outputs[self.L0P0])
            self.outputSums[self.L0P1] = self.activationFunction(self.outputs[self.L0P1])
            self.outputSums[self.L0P2] = self.activationFunction(self.outputs[self.L0P2])
            self.outputs[self.L1P0] = []
            self.outputs[self.L1P1] = []
            for L1fposition in range(0, features.shape[1]):
                for inputNode in [self.L0P0, self.L0P1, self.L0P2]:
                    self.outputs[self.L1P0].append(
                        self.L1P0.weights[L1fposition] * self.outputSums[inputNode] + self.L1P0.bias)
                    self.outputs[self.L1P1].append(
                        self.L1P1.weights[L1fposition] * self.outputSums[inputNode] + self.L1P1.bias)
            self.outputSums[self.L1P0] = self.activationFunction(self.outputs[self.L1P0])
            self.outputSums[self.L1P1] = self.activationFunction(self.outputs[self.L1P1])
            self.outputs[self.L2P0] = []
            for L2position in range(0, features.shape[1]):
                for hiddenNode in [self.L1P0, self.L1P1]:
                    self.outputs[self.L2P0].append(
                        self.L2P0.weights[L2position] * self.outputSums[hiddenNode] + self.L2P0.bias)
            self.outputSums[self.L2P0] = self.activationFunction(self.outputs[self.L2P0])

            """
            #CHOOSING NOT TO REACH STOPPING CONDITION TO TO CLOSENESS OF VALUES
            if prevStdError is None:
                stdError = float(self.outputSums[self.L2P0] - trainingLabel) ** 2
                prevStdError = stdError
            else:
                prevStdError = stdError
                stdError = float(self.outputSums[self.L2P0] - trainingLabel) ** 2
                print('prevStdError is {} and stdError is {}'.format(prevStdError, stdError))
                if stdError < 0.00001 and abs(stdError - prevStdError) < 0.00000000001:
                    return
                else:
                    pass
            """

            # backpropagation
            for fposition in range(0, features.shape[1]):
                self.L2P0.weights[fposition] = self.L2P0.weights[fposition] - (
                learningRate * (trainingLabel - self.outputSums[self.L2P0]) * self.outputSums[self.L2P0] \
                * (1.000000 - self.outputSums[self.L2P0]) * (self.L1P0.weights[fposition] * trainingInstance[fposition] \
                                                             + self.L1P1.weights[fposition] * trainingInstance[
                                                                 fposition]))
                deltaL2P0 = (self.outputSums[self.L2P0] - (
                self.L2P0.weights[fposition] * (self.outputSums[self.L1P0] + self.outputSums[self.L1P1]))) * \
                            self.outputSums[self.L2P0] * (1.000000 - self.outputSums[self.L2P0])
                self.L2P0.bias -= deltaL2P0
                deltaL1P0 = deltaL2P0 * self.outputSums[self.L1P0] * (1.000000 - self.outputSums[self.L1P0])
                deltaL1P1 = deltaL2P0 * self.outputSums[self.L1P1] * (1.000000 - self.outputSums[self.L1P1])
                deltaL1 = deltaL1P0 + deltaL1P1
                self.L1P0.weights[fposition] = self.L1P0.weights[fposition] - (
                learningRate * deltaL1 * self.outputSums[self.L1P0])
                self.L1P0.bias -= deltaL1P0
                self.L1P1.weights[fposition] = self.L1P1.weights[fposition] - (
                learningRate * deltaL1 * self.outputSums[self.L1P0])
                self.L1P1.bias -= deltaL1P1
                deltaL0P0 = deltaL1 * self.outputSums[self.L0P0] * (1.00000 - self.outputSums[self.L0P0])
                deltaL0P1 = deltaL1 * self.outputSums[self.L0P1] * (1.00000 - self.outputSums[self.L0P1])
                deltaL0P2 = deltaL1 * self.outputSums[self.L0P2] * (1.00000 - self.outputSums[self.L0P2])
                deltaL0 = deltaL0P0 + deltaL0P1 + deltaL0P2
                self.L0P0.weights[fposition] = self.L0P0.weights[fposition] - (
                learningRate * deltaL0 * self.outputSums[self.L0P0])
                self.L0P0.bias -= deltaL0P0
                self.L0P1.weights[fposition] = self.L0P1.weights[fposition] - (
                learningRate * deltaL0 * self.outputSums[self.L0P1])
                self.L0P1.bias -= deltaL0P1
                self.L0P2.weights[fposition] = self.L0P2.weights[fposition] - (
                learningRate * deltaL0 * self.outputSums[self.L0P2])
                self.L0P2.bias -= deltaL0P2



    def predict(self, features):
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        # Run model here
        # Return list/array of predictions where there is one prediction for each set of features
        predictions = []
        for testingInstance in features:
            # find largest vector
            largestScalar = 0
            index = None
            del self.outputs[self.L0P0][:]
            del self.outputs[self.L0P1][:]
            del self.outputs[self.L0P2][:]
            for L0fposition in range(0, features.shape[1]):
                self.outputs[self.L0P0].append(self.L0P0.weights[L0fposition] * testingInstance[
                    L0fposition] + self.L0P0.bias)
                self.outputs[self.L0P1].append(self.L0P1.weights[L0fposition] * testingInstance[
                    L0fposition] + self.L0P1.bias)
                self.outputs[self.L0P2].append(self.L0P2.weights[L0fposition] * testingInstance[
                    L0fposition] + self.L0P2.bias)
            self.outputSums[self.L0P0] = self.activationFunction(self.outputs[self.L0P0])
            self.outputSums[self.L0P1] = self.activationFunction(self.outputs[self.L0P1])
            self.outputSums[self.L0P2] = self.activationFunction(self.outputs[self.L0P2])
            del self.outputs[self.L1P0][:]
            del self.outputs[self.L1P1][:]
            for L1fposition in range(0, features.shape[1]):
                for L0P in [self.L0P0, self.L0P1, self.L0P2]:
                    self.outputs[self.L1P0].append(self.L1P0.weights[L1fposition] * self.outputSums[L0P] + self.L1P0.bias)
                    self.outputs[self.L1P1].append(self.L1P1.weights[L1fposition] * self.outputSums[L0P] + self.L1P1.bias)
            self.outputSums[self.L1P0] = self.activationFunction(self.outputs[self.L1P0])
            self.outputSums[self.L1P1] = self.activationFunction(self.outputs[self.L1P1])
            del self.outputs[self.L2P0][:]
            #print('self.outputs[self.L2P0] is of type {} and is {}'.format(type(self.outputs[self.L2P0]), self.outputs[self.L2P0]))
            for L2fposition in range(0, features.shape[1]):
                self.outputs[self.L2P0].append(self.L2P0.weights[L2fposition] * self.outputSums[self.L2P0] + self.L2P0.bias)
                for hiddenNode in [self.L1P0, self.L1P1]:
                    self.outputs[self.L2P0].append(self.L1P0.weights[L2fposition] * self.outputSums[
                        hiddenNode] + self.L1P0.bias)
            self.outputSums[self.L2P0] = self.activationFunction(self.outputs[self.L2P0])
            if self.outputSums[self.L2P0] == 1:
                predictions.append(True)
            else:
                predictions.append(False)

        return predictions



class ID3:
    def __init__(self):
        #Decision tree state here
        #['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
        self.featureOrder = []
        self.featurePositions = [0, 1, 2, 3, 6]
        self.verdicts = [None] * 5

    def entropyMeasure(self, instanceSet, labelSet, featureLocation):
        #entropy of the set
        winners = np.sum(labelSet)
        losers = len(labelSet) - winners
        setEntropy = float(-(winners * math.log(winners, 2)) - (losers * math.log(losers, 2)))
        if featureLocation == 0 or featureLocation == 1:
            above = []
            below = []
            mean = float(sum(instanceSet[featureLocation])) / len(instanceSet)
            for instance, label in zip(instanceSet, labelSet):
                if instance[featureLocation] <= mean:
                    below.append(label)
                else:
                    above.append(label)
            pAbove = float(len(above) / len(labelSet))
            pBelow = float(len(below) / len(labelSet))
            if 0 < sum(above):
                eAbove = float(-(sum(above) * math.log(sum(above), 2.0)) - (len(labelSet) - sum(above) * math.log(len(labelSet) - sum(above), 2.0)))
            else:
                eAbove = 1.0
            if 0 < sum(below):
                eBelow = float(-(sum(below) * math.log(sum(below), 2.0)) - (len(labelSet) - sum(below) * math.log(len(labelSet) - sum(below), 2.0)))
            else:
                eBelow = 1.0
            subSetEntropy = pAbove * eAbove + pBelow * eBelow
            gain = float(setEntropy - subSetEntropy)
            return gain
        elif featureLocation == 2:
            have = []
            haveNot = []
            for instance, label in zip(instanceSet, labelSet):
                if 0 < instance[featureLocation]:
                    haveNot.append(label)
                else:
                    have.append(label)
            pHave = float(len(have) / len(labelSet))
            pHaveNot = float(len(haveNot) / len(labelSet))
            eHave = float(-(sum(have) * math.log(sum(have), 2.0)) - (len(labelSet) - sum(have) * math.log(len(labelSet) - sum(have), 2.0)))
            eHaveNot = float(-(sum(haveNot) * math.log(sum(haveNot), 2.0)) - ((len(labelSet) - sum(haveNot) * math.log(len(labelSet) - sum(haveNot), 2.0))))
            subSetEntropy = pHave * eHave + pHaveNot * eHaveNot
            gain = float(setEntropy - subSetEntropy)
            return gain
        elif featureLocation == 3 or featureLocation == 6:
            cat1 = []
            cat2 = []
            cat3 = []
            for instance, label in zip(instanceSet, labelSet):
                if 0 < instance[featureLocation]:
                    cat1.append(label)
                elif 0 < instance[featureLocation + 1]:
                    cat2.append(label)
                elif 0 < instance[featureLocation + 2]:
                    cat3.append(label)
                else:
                    print('Looking at office, no suitable entry found.')

            pCat1 = float(len(cat1)) / len(labelSet)
            pCat2 = float(len(cat2)) / len(labelSet)
            pCat3 = float(len(cat3)) / len(labelSet)

            if 0 < sum(cat1):
                eCat1 = float(-(sum(cat1) * math.log(sum(cat1), 2.0)) - (len(labelSet) - sum(cat1) * math.log(len(labelSet) - sum(cat1), 2.0)))
            else:
                eCat1 = 1.0
            if 0 < sum(cat2):
                eCat2 = float(-(sum(cat2) * math.log(sum(cat2), 2.0)) - (len(labelSet) - sum(cat2) * math.log(len(labelSet) - sum(cat2), 2.0)))
            else:
                eCat2 = 1.0
            if 0 < sum(cat3):
                eCat3 = float(-(sum(cat3) * math.log(sum(cat3), 2.0)) - (len(labelSet) - sum(cat3) * math.log(len(labelSet) - sum(cat3), 2.0)))
            else:
                eCat3 = 1.0

            subSetEntropy = pCat1 * eCat1 + pCat2 * eCat2 + pCat3 * eCat3
            gain = float(setEntropy - subSetEntropy)
            return gain
        else:
            print('sent improper feature location to entropy function...')


    def train(self, features, labels):
        #training logic here
        #input is list/array of features and labels
        infoGains = []
        if len(features) == 0:
            return
        else:
            self.treeFunc(features, labels, self.featurePositions)


    def treeFunc(self, features, labels, featureLocationList):
        if len(self.featurePositions) < 2:
            self.decider(features, self.featurePositions[0], labels)
        elif len(features) < 5:
            return
        else:
            infoGains = []
            for fLocation in featureLocationList:
                infoGains.append([self.entropyMeasure(features, labels, fLocation), fLocation])
            infoGainFeatureLocation = max(infoGains, key=lambda x: x[0])[1]
            self.featureOrder.append(infoGainFeatureLocation)
            if infoGainFeatureLocation < 3:
                subset1, labels1, subset2, labels2 = self.split(features, infoGainFeatureLocation, labels)
                self.featurePositions.remove(infoGainFeatureLocation)
                self.treeFunc(subset1, labels1, featureLocationList)
                self.treeFunc(subset2, labels2, featureLocationList)
            else:
                subset1, labels1, subset2, labels2, subset3, labels3 = self.split(features, infoGainFeatureLocation, labels)
                self.featurePositions.remove(infoGainFeatureLocation)
                self.treeFunc(subset1, labels1, featureLocationList)
                self.treeFunc(subset2, labels2, featureLocationList)
                self.treeFunc(subset3, labels3, featureLocationList)


    def split(self, features, featureLocation, labels=None):
        if featureLocation == 0 or featureLocation == 1:
            aboveFeatureSet = []
            belowFeatureSet = []
            aboveLabels = []
            belowLabels = []
            mean = float(sum(features[featureLocation])) / len(features)
            for instance, label in zip(features, labels):
                if instance[featureLocation] <= mean:
                    belowLabels.append(label)
                    belowFeatureSet.append(instance)
                else:
                    aboveLabels.append(label)
                    aboveFeatureSet.append(instance)
            return aboveFeatureSet, aboveLabels, belowFeatureSet, belowLabels

        elif featureLocation == 2:
            haveFeatureSet = []
            haveLabels = []
            haveNotFeatureSet = []
            haveNotLabels = []
            for instance, label in zip(features, labels):
                if 0 < instance[featureLocation]:
                    haveNotLabels.append(label)
                    haveNotFeatureSet.append(instance)
                else:
                    haveLabels.append(label)
                    haveFeatureSet.append(instance)
            return haveFeatureSet, haveLabels, haveNotFeatureSet, haveNotLabels

        elif featureLocation == 3 or featureLocation == 6:
            cat1FeatureSet = []
            cat1LabelSet = []
            cat2FeatureSet = []
            cat2LabelSet = []
            cat3FeatureSet = []
            cat3LabelSet = []
            for instance, label in zip(features, labels):
                if 0 < instance[featureLocation]:
                    cat1LabelSet.append(label)
                    cat1FeatureSet.append(instance)
                elif 0 < instance[featureLocation + 1]:
                    cat2LabelSet.append(label)
                    cat2FeatureSet.append(instance)
                elif 0 < instance[featureLocation + 2]:
                    cat3LabelSet.append(label)
                    cat3FeatureSet.append(instance)
                else:
                    print('Looking at office, no suitable entry found.')
            return cat1FeatureSet, cat1LabelSet, cat2FeatureSet, cat2LabelSet, cat3FeatureSet, cat3LabelSet

    def decider(self, features, featureLocation, labels=None):
        if featureLocation == 0 or featureLocation == 1:
            if labels is not None:
                self.verdicts[featureLocation] = [0, 0, 0, 0, 0]
                bucketNumber = 5.0
                rangeIncrement = largestValue / bucketNumber
                for instance, label in zip(features, labels):
                    if instance[featureLocation] <= rangeIncrement:
                        if label is True:
                            self.verdicts[featureLocation][0] += 1
                        else:
                            self.verdicts[featureLocation][0] -= 1
                    elif rangeIncrement < instance[featureLocation] <= 2.0 * rangeIncrement:
                        if label is True:
                            self.verdicts[featureLocation][1] += 1
                        else:
                            self.verdicts[featureLocation][1] -= 1
                    elif 2.0 * rangeIncrement < instance[featureLocation] <= 3.0 * rangeIncrement:
                        if label is True:
                            self.verdicts[featureLocation][2] += 1
                        else:
                            self.verdicts[featureLocation][2] -= 1
                    elif 3.0 * rangeIncrement < instance[featureLocation] <= 4.0 * rangeIncrement:
                        if label is True:
                            self.verdicts[featureLocation][3] += 1
                        else:
                            self.verdicts[featureLocation][3] -= 1
                    else:
                        if label is True:
                            self.verdicts[featureLocation][4] += 1
                        else:
                            self.verdicts[featureLocation][4] -= 1
            else:
                predictions = []
                largestValue = max(features[featureLocation])
                bucketNumber = 5.0
                rangeIncrement = largestValue / bucketNumber
                for instance in features:
                    if instance[featureLocation] <= rangeIncrement:
                        predictions.append(self.verdicts[featureLocation][0])
                    elif rangeIncrement < instance[featureLocation] <= 2.0 * rangeIncrement:
                        predictions.append(self.verdicts[featureLocation][1])
                    elif 2.0 * rangeIncrement < instance[featureLocation] <= 3.0 * rangeIncrement:
                        predictions.append(self.verdicts[featureLocation][2])
                    elif 3.0 * rangeIncrement < instance[featureLocation] <= 4.0 * rangeIncrement:
                        predictions.append(self.verdicts[featureLocation][3])
                    else:
                        predictions.append(self.verdicts[featureLocation][4])
                return predictions

        elif featureLocation == 2:
            if labels is not None:
                self.verdicts[featureLocation] = [0, 0]
                for instance, label in zip(features, labels):
                    print('instance is {}'.format(instance))
                    print('LABEL is {}'.format(label))
                    if 0 < instance[featureLocation]:
                        if label is True:
                            self.verdicts[featureLocation][0] += 1
                        else:
                            self.verdicts[featureLocation][0] -= 1
                    else:
                        if label is True:
                            self.verdicts[featureLocation][1] += 1
                        else:
                            self.verdicts[featureLocation][1] -= 1
            else:
                predictions = []
                for instance in features:
                    if 0 < instance[featureLocation]:
                        predictions.append(self.verdicts[featureLocation][0])
                    else:
                        predictions.append(self.verdicts[featureLocation][1])

                return predictions

        elif featureLocation == 3 or featureLocation == 6:
            if labels is not None:
                self.verdicts[featureLocation] = [0, 0, 0]
                for instance, label in zip(features, labels):
                    if 0 < instance[featureLocation]:
                        if label is True:
                            self.verdicts[featureLocation][0] += 1
                        else:
                            self.verdicts[featureLocation][0] -= 1
                    elif 0 < instance[featureLocation + 1]:
                        if label is True:
                            self.verdicts[featureLocation][1] += 1
                        else:
                            self.verdicts[featureLocation][1] -= 1
                    elif 0 < instance[featureLocation + 2]:
                            if label is True:
                                self.verdicts[featureLocation][2] += 1
                            else:
                                self.verdicts[featureLocation][2] -= 1
            else:
                for instance in features:
                    if 0 < instance[featureLocation]:
                        predictions.append(self.verdicts[featureLocation][0])
                    elif 0 < instance[featureLocation + 1]:
                        predictions.append(self.verdicts[featureLocation][1])
                    elif 0 < instance[featureLocation + 2]:
                        predictions.append(self.verdicts[featureLocation][2])
                return predictions

        else:
            print('sent improper feature location to decider function...')


    def predict(self, features):
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        print('self.featPositions is {}'.format(self.featurePositions))
        predictions = self.decider(features, self.featurePositions[0])
        return predictions



#variables for all classes#
baseFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
hotEncFeatures = ['net_ope_exp', 'net_con', 'tot_loa', 'can_off_P', 'can_off_S', 'can_off_H', \
                  'can_inc_cha_ope_sea_INCUMBENT', 'can_inc_cha_ope_sea_CHALLENGER', 'can_inc_cha_ope_sea_OPEN']
nDataSet = normalizeData(dataset, baseFeatures[:3])
encNormDataset = encodeData(nDataSet, baseFeatures[3:])
encRawDataSet = encodeData(dataset, baseFeatures[3:])


# KNN
KNNRatio = 0.1
KNNtrainingData, KNNtestingData = trainingTestData(encNormDataset, KNNRatio)
KNNtrainingFeatures, KNNtrainingLabels = getNumpy(KNNtrainingData)
KNNtestingFeatures, KNNtestingLabels = getNumpy(KNNtestingData)

K = KNN()
K.train(KNNtrainingFeatures, KNNtrainingLabels)
predictions = K.predict(KNNtestingFeatures)
accuracy = evaluate(predictions, KNNtestingLabels)
print('accuracy for KNN is {}%'.format(accuracy))


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

# MLP
MLPRatio = 0.8
MLPtrainingData, MLPtestingData = trainingTestData(encRawDataSet, MLPRatio)
MLPtrainingFeatures, MLPtrainingLabels = getNumpy(MLPtrainingData)
MLPtestingFeatures, MLPtestingLabels = getNumpy(MLPtestingData)
M = MLP()
M.train(MLPtrainingFeatures, MLPtrainingLabels)
MLPtestingLabels = M.predict(MLPtestingFeatures)
accuracy = evaluate(MLPtestingLabels, MLPtestingLabels)
print('accuracy for MLP is {}%'.format(accuracy))


#ID3
ID3Ratio = 0.7
ID3trainingData, ID3testingData = trainingTestData(encNormDataset, ID3Ratio)
ID3trainingFeatures, ID3trainingLabels = getNumpy(ID3trainingData)
ID3testingFeatures, ID3testingLabels = getNumpy(ID3testingData)

ID3Tree = ID3()
ID3Tree.train(ID3trainingFeatures, ID3trainingLabels)
print('ID3Trees verdict object {}'.format(ID3Tree.verdicts))
ID3testingLabels = ID3Tree.predict(ID3testingFeatures)
accuracy = evaluate(ID3testingLabels, ID3testingLabels)
print('accuracy for ID3 is {}%'.format(accuracy))