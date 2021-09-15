from algorithms import *


def knncollect(data):
    results = []
    for k in range(1, 6):
        param_results = []
        for x in range(50):
            param_results.append(KNN(data, k))
        results.append(param_results)
    return results


def gaussiancollect(data):
    result = []
    floatlist = [1e-9, 1e-5, 1e-1]
    for k in floatlist:
        param_results = []
        for x in range(84):
            param_results.append(Gaussian(data, k))
        result.append(param_results)
    return result


def logregcollect(data):
    result = []
    doublelist = [0.1, 0.5, 1.0, 2.0, 5.0]
    for k in doublelist:
        param_results = []
        for x in range(50):
            param_results.append(LogRegression(data, k))
        result.append(param_results)
    return result


def treecollect(data):
    result = []
    for k in range(1, 11):
        param_results = []
        for x in range(25):
            param_results.append(DT(data, k))
        result.append(param_results)
    return result


def gradientcollect(data):
    result = []
    for k in range(1, 11):
        param_results = []
        for x in range(25):
            param_results.append(GradientDT(data, k))
        result.append(param_results)
    return result

def forestcollect(data):
    result = []
    for k in range(1, 11):
        param_results = []
        for x in range(25):
            param_results.append(Forest(data, k))
        result.append(param_results)
    return result

def mlpcollect(data):
    result = []
    listmlp = [1e-5, 1e-3, 0.1, 10.0]
    for k in listmlp:
        param_results = []
        for x in range(63):
            param_results.append(MLP(data, k))
        result.append(param_results)
    return result

def gaussiancollectPart2(data):
    result = []
    for i in range(100):
        result.append(GaussianPart2(data))
    return result