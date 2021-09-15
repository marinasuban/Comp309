# This is a sample Python script.
import os
import glob
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.cluster import KMeans
from collectors import *
from util import SubPlot, TableSummary, ScatterPlot
import time

def Load(dataset):
    results = [knncollect(dataset),
               gaussiancollect(dataset),
               logregcollect(dataset),
               treecollect(dataset),
               gradientcollect(dataset),
               forestcollect(dataset),
               mlpcollect(dataset)]

    return results

def LoadPart2(dataset, dataset_w_cluster):
    results = [gaussiancollectPart2(dataset),
               gaussiancollectPart2(dataset_w_cluster)]
    return results

# making a fake data set here.
def FakeData():
    # making a fake data set here.
    X, y = make_classification(n_features=20, n_redundant=0, n_informative=5, n_clusters_per_class=1)
    X += 4.0 * np.random.uniform(size=X.shape)
    myfakedataset = (X, y)
    return myfakedataset


def GetData():
    banknote_path = glob.glob(os.getcwd() + "\\data\\banknote-authentication.csv")
    steel_plate_fault_path = glob.glob(os.getcwd() + "\\data\\steel-plate-fault.csv")
    ionosphere_path = glob.glob(os.getcwd() + "\\data\\ionosphere.csv")

    a = pd.read_csv(steel_plate_fault_path[0])
    b = pd.read_csv(ionosphere_path[0])
    c = pd.read_csv(banknote_path[0])

    aX = a.iloc[:, :-1].values  # steel-plate-faults
    aY = a.iloc[:, -1:].values

    bX = b.iloc[:, :-1].values  # ionosphere
    bY = b.iloc[:, -1:].values

    cX = c.iloc[:, :-1].values  # banknote
    cY = c.iloc[:, -1:].values

    return [[aX, aY], [bX, bY], [cX, cY]]

def GetDataPart2():
    iris_path = glob.glob(os.getcwd() + "\\data\\iris.csv")
    banknote_path = glob.glob(os.getcwd() + "\\data\\banknote-authentication.csv")

    a = pd.read_csv(iris_path[0])
    b = pd.read_csv(banknote_path[0])

    aX = a.iloc[:, :-1].values # iris data
    aY = a.iloc[:, -1:].values # class

    bX = b.iloc[:, :-1].values  # banknotes data
    bY = b.iloc[:, -1:].values  # class

    aX = StandardScaler().fit_transform(aX)
    bX = StandardScaler().fit_transform(bX)

    return [[aX, aY], [bX, bY]]

def GenerateDatasetWithClusterLabelsPart2(dataset):
    newX = []
    X = dataset[0]
    y = dataset[1]
    clusterLabels = KMeans(n_clusters=3).fit(X, y).labels_

    for i, x in enumerate(X): # each row
        x_copy = X[i].copy().tolist() # new cloned list
        x_copy.append(clusterLabels[i])
        newX.append(x_copy)

    return [newX, y]

if __name__ == '__main__':
    # part 1
    startTime = time.time()
    x = Process(FakeData())
    a, b, c = GetData()
    result = [Load(Process(a)),  # [Dataset 1][Learning Mode][value][Row][Individual Value]
              Load(Process(b)),  #
              Load(Process(c)),  #
              Load(x)]  #
    print(result);
    SubPlot(result)
    TableSummary(result)
    endTime = time.time()
    print("time elapsed for part 1: ")
    print(endTime - startTime)


    # part 2
    irisData, bankNoteData = GetDataPart2()

    #duplicate X's with a cluster
    irisData_cluster = GenerateDatasetWithClusterLabelsPart2(irisData)
    bankNoteData_cluster = GenerateDatasetWithClusterLabelsPart2(bankNoteData)

    irisResults = LoadPart2(irisData, irisData_cluster)
    bankNoteResults = LoadPart2(bankNoteData, bankNoteData_cluster)

    resultP2 = [irisResults, bankNoteResults]

    ScatterPlot(resultP2)
    plt.show()
