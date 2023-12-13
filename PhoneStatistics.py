import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import matplotlib.cm as cm


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


# =========== Metoda K-średnich =================

# Metoda lokcia

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

# Metoda profilu

# Próba różnych wartości liczby klastrów (od 2 do 6)
n_clusters_range = range(2, 7)
silhouette_scores = []

for n_clusters in n_clusters_range:
    # Tworzenie instancji k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(standarized_array)

    # Obliczanie współczynnika silhouette dla każdego punktu
    silhouette_avg = silhouette_score(standarized_array, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    # Obliczanie wartości współczynnika silhouette dla poszczególnych klastrów
    sample_silhouette_values = silhouette_samples(standarized_array, cluster_labels)

    # Wykres dla każdego klastra
    y_lower = 10
    for i in range(n_clusters):
        # Zebranie wartości współczynnika silhouette dla punktów należących do klastra i posortowanie ich
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.get_cmap("Spectral")(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.title("Wizualizacja współczynnika silhouette dla różnych klastrów")
    plt.xlabel("Wartości współczynnika silhouette")
    plt.ylabel("Numer klastra")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])  
    plt.show()

# Wykres wartości współczynnika silhouette dla różnych liczby klastrów
plt.figure(figsize=(8, 6))
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title('Metoda profilu (Silhouette Method)')
plt.xlabel('Liczba klastrów')
plt.ylabel('Średni współczynnik silhouette')
plt.grid()
plt.show()



# ============== Metoda K-Medoid =======================

# Assuming 'standarized_array' is your standardized data

# Elbow Method
wcss = []
for i in range(1, 11):
    kmedoids = KMedoids(n_clusters=i, init='k-medoids++', max_iter=300, random_state=0)
    kmedoids.fit(standarized_array)
    wcss.append(kmedoids.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Metoda łokcia (Elbow Method) for K-Medoids')
plt.xlabel('Liczba klastrów')
plt.ylabel('Odległość wewnątrz klastrów (WCSS)')
plt.grid()
plt.show()




# Silhouette Method
n_clusters_range = range(2, 7)
silhouette_scores = []

for n_clusters in n_clusters_range:
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmedoids.fit_predict(standarized_array)

    silhouette_avg = silhouette_score(standarized_array, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    # Additional detailed silhouette plot for each number of clusters can be created as well
    # similar to what you've done for KMeans

# Plotting the Silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title('Metoda profilu (Silhouette Method) for K-Medoids')
plt.xlabel('Liczba klastrów')
plt.ylabel('Średni współczynnik silhouette')
plt.grid()
plt.show()






n_clusters_range = range(2, 7)

for n_clusters in n_clusters_range:
    # Create a KMedoids instance
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmedoids.fit_predict(standarized_array)

    # Calculate the average silhouette score
    silhouette_avg = silhouette_score(standarized_array, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg}")

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(standarized_array, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title(f"The silhouette plot for the various clusters with n_clusters = {n_clusters}")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

    #++++++++====================================== zamien X na dane
    # Inicjalizacja macierzy odległości
n_samples = X.shape[0]
distance_matrix = np.zeros((n_samples, n_samples))

# Obliczenie odległości euklidesowej między wszystkimi parami punktów
for i in range(n_samples):
    for j in range(n_samples):
        distance_matrix[i, j] = euclidean(X[i], X[j])

print("Macierz odległości:")
print(distance_matrix)