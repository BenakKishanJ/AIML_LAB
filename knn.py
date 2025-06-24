import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)
        for x in range(1, len(dataSet)): # Start from 1 to skip header if any
            currentInstance = [] # Create a new list for the current instance
            # Convert feature columns to float, keep the last column (class label) as string
            for y in range(len(dataSet[x]) - 1): # Iterate through all but the last column (class label)
                currentInstance.append(float(dataSet[x][y]))
            currentInstance.append(dataSet[x][-1]) # Append the last column (class label) as string

            if random.random() < split:
                trainingSet.append(currentInstance)
            else:
                testSet.append(currentInstance)

def euclideanDistance(instance1, instance2, length):
    distance=0
    for x in range(length):
        distance+=(instance1[x]-instance2[x])**2
    return math.sqrt(distance)

def getNeighbors(trainingSet, TestInstance, k):
    distances=[]
    length=len(TestInstance)-1 # Exclude the class label from distance calculation
    for x in range(len(trainingSet)):
        dist=euclideanDistance(TestInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1)) # Sort by distance
    neighbors=[]
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1] # Get the class label of the neighbor
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response]=1
    sortedVotes=sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0] # Return the class with the most votes


def getAccuracy(testSet, predictions):
    correct=0
    for x in range(len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1

    return (correct/float(len(testSet)))*100

def main():
    trainingSet=[]
    testSet=[]
    split=0.67 # Percentage of data for training
    loadDataset("dataset/iris.csv", split, trainingSet, testSet)
    print("\n Number of training instances: ", len(trainingSet))
    print(" Number of test instances: ", len(testSet))

    # Generate predictions
    predictions=[]
    k=3 # Number of neighbors
    print("\n General predictions:")
    for x in range(len(testSet)):
        neighbors=getNeighbors(trainingSet, testSet[x], k)
        result= getResponse(neighbors)
        predictions.append(result)

    # Print individual predictions and actuals
    for x in range(len(testSet)):
        print('predicted: ', repr(predictions[x]), ' actual: ', repr(testSet[x][-1]))

    # Calculate and print overall accuracy
    accuracy=getAccuracy(testSet, predictions)
    print(' Accuracy: ', repr(accuracy), '%')

if __name__=="__main__":
    main()
