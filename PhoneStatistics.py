import pandas as pd
import numpy as np

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

readData=pd.read_csv("DaneTelefonow.csv",sep=";")
#readData=pd.read_csv("C:/Users/zapar/Python/BOT/Statystyczna-analiza-danych/DaneTelefonow.csv",sep=";")
numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()

print(numpy_array2)
numpy_array2 = ChangeVariablesToStimulants(numpy_array2,[1,4,7])
print('A to zmienione dane')
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

    
    

    
    