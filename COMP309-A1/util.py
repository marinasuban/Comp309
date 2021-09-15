import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from common import *


def Merge(list):
    mergeList = []
    for m in list:
        for n in m:
            mergeList.append(n)
    return mergeList


def HighestMeanDataset(dataset):
    results = []
    for algorithm in dataset:
        [index, value] = HighestMean(algorithm)
        results.append([index, value])

    results[0][0] = KNNValues[results[0][0]]  # get KNN strings
    results[1][0] = GaussianNBValues[results[1][0]]  # get Gaussian strings
    results[2][0] = LogisticRegressionValues[results[2][0]]  # etc
    results[3][0] = DTValues[results[3][0]]  #
    results[4][0] = GradientBoostingValues[results[4][0]]  #
    results[5][0] = RandomForestValues[results[5][0]]  #
    results[6][0] = MLPValues[results[6][0]]  #
    return results


def HighestMean(list):
    if isinstance(list, int):
        return [1, 9999]
    valuemean = []
    for h in list:
        valuemean.append(np.mean(h))

    maxMean = -1
    maxIndex = -1
    for index, item in enumerate(valuemean):
        if item > maxMean:
            maxIndex = index
            maxMean = item

    return [maxIndex, maxMean]


def SubPlot(result):

    rows = 4
    columns = 7

    # results[x][y] = [dataset][algorithm]
    fig, axs = plt.subplots(rows, columns, figsize=(16, 9))
    fig.text(0.04, 0.5, 'Accuracy %', va='center', rotation='vertical')
    setBoxPlot(axs, columns, rows, result)

    axs[0, 0].set_title('KNearestNeighbour')
    axs[0, 1].set_title('GaussianNB')
    axs[0, 2].set_title('LogisticRegression')
    axs[0, 3].set_title('DecisionTree')
    axs[0, 4].set_title('GradientBoosting')
    axs[0, 5].set_title('RandomForest')
    axs[0, 6].set_title('MLP')

    axs[0, 0].set_ylabel('steel plate faults')
    axs[1, 0].set_ylabel('ionosphere')
    axs[2, 0].set_ylabel('banknote authentication')
    axs[3, 0].set_ylabel('generated data')

    setXLabels(axs, 0, KNNValues) # knn
    setXLabels(axs, 1, GaussianNBValues) # gaussian
    setXLabels(axs, 2, LogisticRegressionValues) # logreg
    setXLabels(axs, 3, DTValues) # Dtree
    setXLabels(axs, 4, GradientBoostingValues)  # Gradient
    setXLabels(axs, 5, RandomForestValues)  # Forest
    setXLabels(axs, 6, MLPValues)  # MLP

def setBoxPlot(axs, column, row, result):
    for x in range(row):
        for y in range(column):
            axs[x, y].boxplot(result[x][y])

    # # KNN
    # fig, axs = plt.subplots()
    # fig.suptitle('KNeighborsClassifier')
    # knnlist = [Merge(result[0][0]), Merge(result[1][0]), Merge(result[2][0]), Merge(result[3][0])]
    # axs.boxplot(knnlist)
    # axs.set(ylabel='Accuracy', xlabel='1 = steel, 2 = ionosphere, 3=banknote, 4=generated')
    #
    # # GNB
    # fig1, axs1 = plt.subplots()
    # fig1.suptitle('GaussianNB')
    # gnblist = [Merge(result[0][1]), Merge(result[1][1]), Merge(result[2][1]), Merge(result[3][1])]
    # axs1.boxplot(gnblist)
    # axs1.set(ylabel='Accuracy', xlabel='1 = steel, 2 = ionosphere, 3=banknote, 4=generated')
    #
    # # LR
    # fig2, axs2 = plt.subplots()
    # fig2.suptitle('LogisticRegression')
    # lrlist = [Merge(result[0][2]), Merge(result[1][2]), Merge(result[2][2]), Merge(result[3][2])]
    # axs2.boxplot(lrlist)
    # axs2.set(ylabel='Accuracy', xlabel='1 = steel, 2 = ionosphere, 3=banknote, 4=generated')
    #
    # # DT
    # fig3, axs3 = plt.subplots()
    # fig3.suptitle('DecisionTreeClassifier')
    # dtlist = [Merge(result[0][3]), Merge(result[1][3]), Merge(result[2][3]), Merge(result[3][3])]
    # axs3.boxplot(dtlist)
    # axs3.set(ylabel='Accuracy', xlabel='1 = steel, 2 = ionosphere, 3=banknote, 4=generated')
    #
    # # GDT
    # fig4, axs4 = plt.subplots()
    # fig4.suptitle('GradientBoostingClassifier')
    # gdtlist = [Merge(result[0][4]), Merge(result[1][4]), Merge(result[2][4]), Merge(result[3][4])]
    # axs4.boxplot(gdtlist)
    # axs4.set(ylabel='Accuracy', xlabel='1 = steel, 2 = ionosphere, 3=banknote, 4=generated')
    #
    # # RFC
    # fig5, axs5 = plt.subplots()
    # fig5.suptitle('RandomForestClassifier')
    # rfclist = [Merge(result[0][5]), Merge(result[1][5]), Merge(result[2][5]), Merge(result[3][5])]
    # axs5.boxplot(rfclist)
    # axs5.set(ylabel='Accuracy', xlabel='1 = steel, 2 = ionosphere, 3=banknote, 4=generated')
    #
    # # MLPC
    # fig6, axs6 = plt.subplots()
    # fig6.suptitle('MLPClassifier')
    # mlpclist = [Merge(result[0][6]), Merge(result[1][6]), Merge(result[2][6]), Merge(result[3][6])]
    # axs6.boxplot(mlpclist)
    # axs6.set(ylabel='Accuracy', xlabel='1 = steel, 2 = ionosphere, 3=banknote, 4=generated')

def setXLabels(axs, column, xticklabels):
    axs[0, column].axes.get_xaxis().set_visible(False)
    axs[1, column].axes.get_xaxis().set_visible(False)
    axs[2, column].axes.get_xaxis().set_visible(False)
    axs[3, column].set_xticklabels(xticklabels, minor=False, rotation=60)

def TableSummary(result):
    #labels
    datasets = ["Steel-plate-faults", "Ionosphere", "Banknote-authentication", "Generated Dataset"]
    tableheaders = ["+", "KNeighborsClassifier", "GaussianNB", "LogisticRegression", "DecisionTreeClassifier", "GradientBoostingClassifier", "RandomForestClassifier", "MLPClassifier"]

    Table1 = PrettyTable(tableheaders)
    Table2 = PrettyTable(tableheaders)
    OverallTable = PrettyTable(tableheaders)
    # Table I is to contain the best average value (of validation error)
    # Table II is to contain the associated (ie. best) value for the control parameter.
    for i in range(0, 4):  # 4 datasets
        table_1_row_data = [datasets[i]]
        table_2_row_data = [datasets[i]]
        overall_tb_row_data = [datasets[i]]


        row_values = HighestMeanDataset(result[i])
        for index, value in enumerate(row_values):
            table_1_row_data.append(value[1])  # add the highest mean (best average value) of each algorithm
            table_2_row_data.append(value[0])  # add the associated parameter values for that best average

        overall_tb_dataset = result[i]
        for algorithm in overall_tb_dataset:
            overall_tb_row_data.append(np.mean(algorithm)) #add the mean of all 250 runs for each algorithm

        Table1.add_row(table_1_row_data)
        Table2.add_row(table_2_row_data)
        OverallTable.add_row(overall_tb_row_data)

    print("+------------Table 1--------------+")
    print(Table1)
    print("+------------Table 2--------------+")
    print(Table2)
    print("+------------Overall Averages per algorithm--------------+")
    print(OverallTable)

def ScatterPlot(results):
    fig1, axs1 = plt.subplots()
    fig1.suptitle('IrisData')
    axs1.scatter(results[0][0], results[0][1])
    axs1.plot([0, 1], [0, 1], transform=axs1.transAxes, ls="--", c=".3")
    axs1.set(ylabel='with cluster', xlabel='without cluster')

    fig2, axs2 = plt.subplots()
    fig2.suptitle('BankNotesData')
    axs2.scatter(results[1][0], results[1][1])
    axs2.plot([0, 1], [0, 1], transform=axs2.transAxes, ls="--", c=".3")
    axs2.set(ylabel='with cluster', xlabel='without cluster')
    return
