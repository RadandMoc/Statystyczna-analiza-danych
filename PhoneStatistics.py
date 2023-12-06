import pandas as pd
import numpy as np

print("Hello World")

readData=pd.read_csv("DaneTelefonow.csv",sep=";")
#readData=pd.read_csv("C:/Users/zapar/Python/BOT/Statystyczna-analiza-danych/DaneTelefonow.csv",sep=";")
numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()
print(numpy_array2)



def getOptimalObject(array):
    return np.max(array,0)

def getMeansArray(array):
    return np.mean(array,0)
    
def getStandardDeviationArray(array):
    return np.std(array,0)

def GetStandarizatedArray(array,meanArray,stdArray):
    return (array-meanArray)/stdArray
    
meanArray = getMeansArray(numpy_array2)
stdArray = getStandardDeviationArray(numpy_array2)

GetStandarizatedArray(numpy_array2,meanArray,stdArray)

    
    

    
    