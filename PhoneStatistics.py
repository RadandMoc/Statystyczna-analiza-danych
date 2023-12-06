import pandas as pd
import numpy as np
import math

def ChangeVariablesToStimulants(data,numbersOfDestimulants):
    if not isinstance(data, np.ndarray):
        raise ValueError("Podane dane nie są typu NumPy array")
    
    if not isinstance(numbersOfDestimulants, list):
        raise ValueError("Numer kolumn musi być w postaci listy")
    
    num_columns = data.shape[1]
    for col_num in numbersOfDestimulants:
        if col_num >= num_columns:
            raise ValueError(f"Kolumna numer {col_num} nie istnieje w danych")
        
    data[:, numbersOfDestimulants] *= -1
    return data

def CalculateMeanOfArray(array):
    if not isinstance(array, np.ndarray):
        raise ValueError("Podane dane nie są typu NumPy array")
    
    if array.ndim != 1:
        raise ValueError("Dane powinny być jednowymiarową tablicą NumPy")
    
    mean = np.mean(array)
    return mean

def CalculateStandardDeviation(array):
    if not isinstance(array, np.ndarray):
        raise ValueError("Podane dane nie są typu NumPy array")
    
    if array.ndim != 1:
        raise ValueError("Dane powinny być jednowymiarową tablicą NumPy")
    
    standardDeviation = np.std(array)
    return standardDeviation

def DistanceAsFarAsPossible(distance):
    returner = CalculateMeanOfArray(distance) + 2 * CalculateStandardDeviation(distance)
    return returner



def getOptimalObject(array):
    return np.max(array,0)

def getMeansArrayOfColumn(array):
    return np.mean(array,0)
    
def getStandardDeviationArrayOfColumn(array):
    return np.std(array,0)

def GetStandarizatedArray(array,meanArray,stdArray):
    return (array-meanArray)/stdArray
    

def getImportanceArray(array,weight):
    return array*we
    
def DistanceFromObjectToMasterObject(array,optimalObject,weight=None):
    if weight is None:
        weight = np.ones(len(array[:,1]))
        
    squaredDistance = (array-optimalObject)**2
    importanceDistance = squaredDistance*weight
    result = importanceDistance.sum(axis=1)
    result = result ** (1/2) 
    return result
   
    
    
def getFinalResult(arrayWithDi,d0):
    return 1-(arrayWithDi/d0)
    
    
    
#readData=pd.read_csv("DaneTelefonow.csv",sep=";")
readData=pd.read_csv("C:/Users/zapar/Python/BOT/Statystyczna-analiza-danych/DaneTelefonow.csv",sep=";")
numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()


numpy_array2 = ChangeVariablesToStimulants(numpy_array2,[1,4,7])


    
meanArray = getMeansArrayOfColumn(numpy_array2)
stdArray = getStandardDeviationArrayOfColumn(numpy_array2)

stddat = GetStandarizatedArray(numpy_array2,meanArray,stdArray)
optimalObject = getOptimalObject(stddat)
weight = np.array([0.17,0.21,0.11,0.1,0.05,0.08,0.065,0.085,0.13])
distance = DistanceFromObjectToMasterObject(stddat,optimalObject,weight)

"""
print(meanArray)
print(stdArray)
print(optimalObject)
git
"""

print("Dystanse"+str(distance)+"")
print("Maksymalny dystans"+str(DistanceAsFarAsPossible(distance))+"")
print(getFinalResult(distance,DistanceAsFarAsPossible(distance)))


    
    