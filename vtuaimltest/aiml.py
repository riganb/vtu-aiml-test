def load_program(program_number: int, print_program: bool = True):
    PROGRAMS = [
        """
def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}
    parents = {}

    g[start_node] = 0
    parents[start_node] = start_node
    while len(open_set) > 0:
        n = None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v
        if n == stop_node or Graph_nodes[n] is None:
            pass
        else:
            for (m, weight) in get_neighbours(n):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)
        if n is None:
            print('path does not exist!')
            return None
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('path found: {}'.format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print('path does not exist!')
    return None


def get_neighbours(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None


def heuristic(n):
    H_dist = {
        'A': 10,
        'B': 8,
        'C': 5,
        'D': 7,
        'E': 3,
        'F': 6,
        'G': 5,
        'H': 3,
        'I': 1,
        'J': 0
    }
    return H_dist[n]


Graph_nodes = {
    'A': [('B', 6), ('F', 3)],
    'B': [('C', 3), ('D', 2)],
    'C': [('D', 1), ('E', 5)],
    'D': [('C', 1), ('E', 8)],
    'E': [('I', 5), ('J', 5)],
    'F': [('G', 1), ('H', 7)],
    'G': [('I', 3)],
    'H': [('I', 2)],
    'I': [('E', 5), ('J', 3)]
}

aStarAlgo('A', 'J')
        """,
        '''
"""Recursive implementation of AO* algorithm"""


class Graph:
    def __init__(self, graph, heuristicNodeList, startNode):  # instantiate graph object with graph topology,
        # heuristic values, start node

        self.graph = graph
        self.H = heuristicNodeList
        self.start = startNode
        self.parent = {}
        self.status = {}
        self.solutionGraph = {}

    def applyAOStar(self):  # starts a recursive AO* algorithm
        self.aoStar(self.start, False)

    def getNeighbors(self, v):  # gets the Neighbors of a given node
        return self.graph.get(v, '')

    def getStatus(self, v):  # return the status of a given node
        return self.status.get(v, 0)

    def setStatus(self, v, val):  # set the status of a given node
        self.status[v] = val

    def getHeuristicNodeValue(self, n):
        return self.H.get(n, 0)  # always return the heuristic value of a given node

    def setHeuristicNodeValue(self, n, value):
        self.H[n] = value  # set the revised heuristic value of a given node

    def printSolution(self):
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START NODE:", self.start)
        print("------------------------------------------------------------")
        print(self.solutionGraph)
        print("------------------------------------------------------------")

    def computeMinimumCostChildNodes(self, v):  # Computes the Minimum Cost of child nodes of a given node v
        minimumCost = 0
        costToChildNodeListDict = {}
        costToChildNodeListDict[minimumCost] = []
        flag = True
        for nodeInfoTupleList in self.getNeighbors(v):  # iterate over all the set of child node/s
            cost = 0
            nodeList = []
            for c, weight in nodeInfoTupleList:
                cost = cost + self.getHeuristicNodeValue(c) + weight
                nodeList.append(c)

            if flag == True:  # initialize Minimum Cost with the cost of first set of child node/s
                minimumCost = cost
                costToChildNodeListDict[minimumCost] = nodeList  # set the Minimum Cost child node/s
                flag = False
            else:  # checking the Minimum Cost nodes with the current Minimum Cost
                if minimumCost > cost:
                    minimumCost = cost
                    costToChildNodeListDict[minimumCost] = nodeList  # set the Minimum Cost child node/s

        return minimumCost, costToChildNodeListDict[minimumCost]  # return Minimum Cost and Minimum Cost child node/s

    def aoStar(self, v, backTracking):  # AO* algorithm for a start node and backTracking status flag

        print("HEURISTIC VALUES  :", self.H)
        print("SOLUTION GRAPH    :", self.solutionGraph)
        print("PROCESSING NODE   :", v)
        print("-----------------------------------------------------------------------------------------")

        if self.getStatus(v) >= 0:  # if status node v >= 0, compute Minimum Cost nodes of v
            minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
            self.setHeuristicNodeValue(v, minimumCost)
            self.setStatus(v, len(childNodeList))

            solved = True  # check the Minimum Cost nodes of v are solved
            for childNode in childNodeList:
                self.parent[childNode] = v
                if self.getStatus(childNode) != -1:
                    solved = solved & False

            if solved == True:  # if the Minimum Cost nodes of v are solved, set the current node status as solved(-1)
                self.setStatus(v, -1)
                self.solutionGraph[
                    v] = childNodeList  # update the solution graph with the solved nodes which may be a part of
                # solution

            if v != self.start:  # check the current node is the start node for backtracking the current node value
                self.aoStar(self.parent[v],
                            True)  # backtracking the current node value with backtracking status set to true

            if not backTracking:  # check the current call is not for backtracking
                for childNode in childNodeList:  # for each Minimum Cost child node
                    self.setStatus(childNode, 0)  # set the status of child node to 0(needs exploration)
                    self.aoStar(childNode,
                                False)  # Minimum Cost child node is further explored with backtracking status as false


h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
graph1 = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]
}
G1 = Graph(graph1, h1, 'A')
G1.applyAOStar()
G1.printSolution()

h2 = {'A': 1, 'B': 6, 'C': 12, 'D': 10, 'E': 4, 'F': 4, 'G': 5, 'H': 7}  # Heuristic values of Nodes
graph2 = {  # Graph of Nodes and Edges
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],  # Neighbors of Node 'A', B, C & D with respective weights
    'B': [[('G', 1)], [('H', 1)]],  # Neighbors are included in a list of lists
    'D': [[('E', 1), ('F', 1)]]  # Each sublist indicate a "OR" node or "AND" nodes
}

G2 = Graph(graph2, h2, 'A')  # Instantiate Graph object with graph, heuristic values and start Node
G2.applyAOStar()  # Run the AO* algorithm
G2.printSolution()  # Print the solution graph as output of the AO* algorithm search
        ''',
        """
import csv

with open("/content/CandidateElimination.csv") as f:
    csv_file = csv.reader(f)
    data = list(csv_file)

    print("Positive Samples are: \n")
    for i in data:
      if i[-1]=="Yes":
        print(i,"\n")
        
    s = data[1][:-1]

    g = [['?' for i in range(len(s))] for j in range(len(s))]

    for i in data:
        if i[-1] == "Yes":
            for j in range(len(s)):
                if i[j] != s[j]:
                    s[j] = '?'
                    g[j][j] = '?'

        elif i[-1] == "No":
            for j in range(len(s)):
                if i[j] != s[j]:
                    g[j][j] = s[j]
                else:
                    g[j][j] = "?"
        print("\nSteps of Candidate Elimination Algorithm", data.index(i) + 1)
        print(s)
        print(g)

    gh = []
    for i in g:
        for j in i:
            if j != '?':
                gh.append(i)
                break
    print("\nFinal specific hypothesis:\n", s)

    print("\nFinal general hypothesis:\n", gh)

        """,
        """
import numpy as np
import pandas as pd

data = pd.read_csv('p4.csv')

def entropy(target):
    val,counts = np.unique(target,return_counts = True)
    ent = 0
    for i in range(len(val)):
        c = counts[i]/sum(counts)
        ent += -c*np.log2(c)
    return ent

def infoGain(data,features,target):
    te = entropy(data[target])
    val,counts = np.unique(data[features],return_counts = True)
    eg = 0
    for i in range(len(val)):
        c = counts[i]/sum(counts)
        eg += c*entropy(data[data[features] == val[i]][target])
    InfoGain = te-eg
    return InfoGain

def id3(data, features, target, pnode):
    
    t = np.unique(data[target])
    
    if len(t) == 1:
        return t[0]
    
    if len(features) == 0:
        return pnode
    
    pnode = t[np.argmax(t[1])]
    
    IG = [infoGain(data,f,target) for f in features]
    index = np.argmax(IG)
    
    col = features[index]
    tree = {col:{}}
    
    features = [f for f in features if f!=col]
    
    for val in np.unique(data[col]):
        sub_data = data[data[col]==val].dropna()
        subtree = id3(sub_data,features,target,pnode)
        tree[col][val] = subtree
    return tree

testData = data.sample(frac = 0.1)
data.drop(testData.index,inplace = True)

target = 'play'
features = data.columns[data.columns!=target]

tree = id3(data,features,target,None)
print (tree, end='\n\n')

test = testData.to_dict('records')[0]
print(test, '=>', id3(test,features,target,None))
        """,
        """
import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X / np.amax(X, axis=0)  # maximum of X array longitudinally
y = y / 100


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


# Variable initialization
epoch = 7000  # Setting training iterations
lr = 0.1  # Setting learning rate
inputlayer_neurons = 2  # number of features in data set
hiddenlayer_neurons = 3  # number of hidden layers neurons
output_neurons = 1  # number of neurons at output layer
# weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))
# draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
    # Forward Propogation
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)
    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    wout += hlayer_act.T.dot(d_output) * lr
    # bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) * lr
    # bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n", output)
        """,
        """
import csv
import random
import math

def loadCsv(filename):
  lines = csv.reader(open(filename, "r"))
  dataset = list(lines)
  for i in range(len(dataset)):
    dataset[i] = [float(x) for x in dataset[i]]
  return dataset

def splitDataset(dataset, splitRatio):
  trainSize = int(len(dataset) * splitRatio)
  trainSet = []
  copy = list(dataset)
  while len(trainSet) < trainSize:
    index = random.randrange(len(copy))
    trainSet.append(copy.pop(index))
  return [trainSet, copy]

def separateByClass(dataset):
  separated = {}
  for i in range(len(dataset)):
    vector = dataset[i]
    if (vector[-1] not in separated):
      separated[vector[-1]] = []
    separated[vector[-1]].append(vector)
  return separated

def mean(numbers):
  return sum(numbers)/float(len(numbers))

def stdev(numbers):
  avg = mean(numbers)
  variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
  return math.sqrt(variance)

def summarize(dataset):
  summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
  del summaries[-1]
  return summaries

def summarizeByClass(dataset):
  separated = separateByClass(dataset)
  summaries = {}
  for classValue, instances in separated.items():
    summaries[classValue] = summarize(instances)
  return summaries

def calculateProbability(x, mean, stdev):
  exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
  return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
  probabilities = {}
  for classValue, classSummaries in summaries.items():
    probabilities[classValue] = 1
    for i in range(len(classSummaries)):
      mean, stdev = classSummaries[i]
      x = inputVector[i]
      probabilities[classValue] *= calculateProbability(x, mean, stdev)
  return probabilities

def predict(summaries, inputVector):
  probabilities = calculateClassProbabilities(summaries, inputVector)
  bestLabel, bestProb = None, -1
  for classValue, probability in probabilities.items():
    if bestLabel is None or probability > bestProb:
      bestProb = probability
      bestLabel = classValue
  return bestLabel

def getPredictions(summaries, testSet):
  predictions = []
  for i in range(len(testSet)):
    result = predict(summaries, testSet[i])
    predictions.append(result)
  return predictions

def getAccuracy(testSet, predictions):
  correct = 0
  for i in range(len(testSet)):
    if testSet[i][-1] == predictions[i]:
      correct += 1
  return (correct/float(len(testSet))) * 100.0

def main():
  filename = 'diabetes.csv'
  splitRatio = 0.87
  dataset = loadCsv(filename)

  trainingSet, testSet = splitDataset(dataset, splitRatio)

  print(len(testSet))

  print('Spliting {} rows into training={} and testing={} rows'.format(len(dataset), len(trainingSet), len(testSet)))
  # prepare model
  summaries = summarizeByClass(trainingSet)
 
  # test model
  predictions = getPredictions(summaries, testSet)
  accuracy = getAccuracy(testSet, predictions)
  print('Classification Accuracy: {}%'.format(accuracy))
  print("-----------------------------------------------------")

main()
        """,
        """
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture 
import sklearn.metrics as sm
import pandas as pd
import numpy as np

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
Y = pd.DataFrame(iris.target)
Y.columns = ['Targets']

print(X)
print(Y)
colormap = np.array(['red', 'lime', 'black'])

plt.subplot(1,2,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[Y.Targets], s=40)
plt.title('Real Clustering')

model1 = KMeans(n_clusters=3)
model1.fit(X)

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model1.labels_], s=40)
plt.title('K Mean Clustering')
plt.show()

model2 = GaussianMixture(n_components=3) 
model2.fit(X)

plt.subplot(1,2,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model2.predict(X)], s=40)
plt.title('EM Clustering')
plt.show()

print("Actual Target is:\n", iris.target)
print("K Means:\n",model1.labels_)
print("EM:\n",model2.predict(X))
print("Accuracy of KMeans is ",sm.accuracy_score(Y,model1.labels_))
print("Accuracy of EM is ",sm.accuracy_score(Y, model2.predict(X)))
        """,
        """
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
print("\nIRIS FEATURES \TARGET NAMES: \n", iris_dataset.target_names)


for i in range(len(iris_dataset.target_names)):
    print("\n[{0}]:[{1}]".format(i, iris_dataset.target_names[i]))
print("\nIRIS DATA :\n", iris_dataset["data"])


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)
print("\nTarget :\n", iris_dataset["target"])
print("\nX TRAIN \n", X_train)
print("\nX TEST \n", X_test)
print("\nY TRAIN \n", y_train)
print("\nY TEST \n", y_test)


kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
print("\nXNEW \n", x_new)
prediction = kn.predict(x_new)

print("\nPredicted target value: {}\n".format(prediction))
print("\nPredicted feature name:{}\n".format(iris_dataset["target_names"][prediction]))


i = 1
x = X_test[i]
x_new = np.array([x])
print("\nXNEW \n", x_new)
for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    print(
        "\n Actual:[{0}][{1}] \t, Predicted:{2}{3}".format(
            y_test[i],
            iris_dataset["target_names"][y_test[i]],
            prediction,
            iris_dataset["target_names"][prediction],
        )
    )
print("\nTEST SCORE[ACCURACY]: {:.2f}\n".format(kn.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

y_pred = kn.predict(X_test)
print("Confusion matrix is: ")
print(confusion_matrix(y_test, y_pred))
        """,
        """
import numpy as np
from ipywidgets import interact
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from bokeh.io import push_notebook
output_notebook()

def local_regression(x0, X, Y, tau):
 x0 = np.r_[1, x0]
 X = np.c_[np.ones(len(X)), X]
 # fit model: normal equations with kernel
 xw = X.T * radial_kernel(x0, X, tau)
 beta = np.linalg.pinv(xw @ X) @ xw @ Y
 # predict value
 return x0 @ beta

def radial_kernel(x0, X, tau):
 return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau))

n = 1000
# generate dataset
X = np.linspace(-3, 3, num=n)
Y = np.log(np.abs(X ** 2 - 1) + .5)
# jitter X
X += np.random.normal(scale=.1, size=n)

def plot_lwr(tau):
 # prediction
 domain = np.linspace(-3, 3, num=300)
 prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
 plot = figure(plot_width=400, plot_height=400)
 plot.title.text = 'tau=%g' % tau
 plot.scatter(X, Y, alpha=.3)
 plot.line(domain, prediction, line_width=2, color='red')
 return plot

show(gridplot([[plot_lwr(10.), plot_lwr(1.)],[plot_lwr(0.1), plot_lwr(0.01)]]))
# #show(plot, notebook_handle=True)
# interact(interactive_update, tau=(0.01, 3., 0.01))
        """,
    ]
    if print_program == False:
        return PROGRAMS[program_number]
    else:
        print(PROGRAMS[program_number])
