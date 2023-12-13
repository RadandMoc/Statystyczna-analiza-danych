import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

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
    return array*weight
    
def DistanceFromObjectToMasterObject(array,optimalObject,weight=None):
    if weight is None:
        weight = np.ones(len(array[:,1]))
        
    squaredDistance = (array-optimalObject)**2
    importanceDistance = squaredDistance*weight
    result = importanceDistance.sum(axis=1)
    result = result ** (1/2) 
    return result
   
def getFinalResult(arrayWithDi,d0):
    return 1 - (arrayWithDi/d0)
 
def getTheWorstObject(array):
    return np.min(array,0)
    
def normalizedMatrixTopsisMethod(array):
    xijSquared = array ** 2
    sumxijSquared = xijSquared.sum(axis=0) #column
    denominator = sumxijSquared ** (1/2)
    return array/denominator
    
def getRiInTopsisMethod(diPlus,diMinus):
    return diMinus/(diPlus + diMinus)
    
    


    
readData=pd.read_csv("DaneTelefonow.csv",sep=";")
#readData=pd.read_csv("C:/Users/zapar/Python/BOT/Statystyczna-analiza-danych/DaneTelefonow.csv",sep=";")

numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()


numpy_array2 = ChangeVariablesToStimulants(numpy_array2,[1,4,7])

"""
Metoda Hellwiga
    
meanArray = getMeansArrayOfColumn(numpy_array2)
stdArray = getStandardDeviationArrayOfColumn(numpy_array2)

stddat = GetStandarizatedArray(numpy_array2,meanArray,stdArray)
optimalObject = getOptimalObject(stddat)
weight = np.array([0.17,0.21,0.11,0.1,0.05,0.08,0.065,0.085,0.13])
distance = DistanceFromObjectToMasterObject(stddat,optimalObject,weight)


print(meanArray)
print(stdArray)
print(optimalObject)
git


print("Dystanse"+str(distance)+"")
print("Maksymalny dystans"+str(DistanceAsFarAsPossible(distance))+"")
finalResult=getFinalResult(distance,DistanceAsFarAsPossible(distance))
finalDataFrame = pd.DataFrame({'Telefony': readData.iloc[:,0], 'Wynik': finalResult})
print(finalDataFrame.sort_values(by='Wynik',ascending=False))
"""

#Metoda Topsis

normalizedMatrix = normalizedMatrixTopsisMethod(numpy_array2)

theBestObject = getOptimalObject(normalizedMatrix)
theWorstObject = getTheWorstObject(normalizedMatrix)

weight = np.array([0.17,0.21,0.11,0.1,0.05,0.08,0.065,0.085,0.13])


#distance to the best object
diPlus = DistanceFromObjectToMasterObject(normalizedMatrix,theBestObject,weight)
#distance to the worst object
diMinus = DistanceFromObjectToMasterObject(normalizedMatrix,theWorstObject,weight)

Ri = getRiInTopsisMethod(diMinus,diPlus)


finalDataFrame = pd.DataFrame({'Telefony': readData.iloc[:,0], 'Wynik': Ri})
print(finalDataFrame.sort_values(by='Wynik'))




#ANALIZA Skupień


# Number of clusters (centers)

numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()

standarized_array = GetStandarizatedArray(numpy_array2,getMeansArrayOfColumn(numpy_array2),getStandardDeviationArrayOfColumn(numpy_array2))

nClusters = 2

# Number of times the k-means algorithm will be run with different centroid seeds (nstart)
nstart = 10

# Create and fit the KMeans model
kmeansMethod = KMeans(n_clusters=nClusters, n_init=nstart, random_state=0).fit(standarized_array)

# Cluster centers
centers = kmeansMethod.cluster_centers_

labels = kmeansMethod.predict(standarized_array)

print(labels)


# Metoda K-metoid
#numpy_array2 - chwilowa dana, później zostaje zmienione na dane jakie Tomek zestandaryzuje
# Próba różnych wartości liczby klastrów (od 1 do 10) i obliczanie wartości odległości
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(standarized_array)
    wcss.append(kmeans.inertia_)

# Wykres metody łokcia (elbow method)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Metoda łokcia (Elbow Method)')
plt.xlabel('Liczba klastrów')
plt.ylabel('Odległość wewnątrz klastrów (WCSS)')
plt.grid()
plt.show()