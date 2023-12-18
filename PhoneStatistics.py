import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import scipy.spatial.distance as ssd
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn_extra.cluster import KMedoids
import matplotlib.cm as cm
from scipy.spatial.distance import euclidean


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
        weight = np.ones(len(array[1,:]))
        
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
#readData=pd.read_csv("DaneTelefonowBezOutsiderow.csv",sep=";")
pd.set_option('display.max_columns', None)  # Wyświetlanie wszystkich kolumn
pd.set_option('display.max_rows', None) 
columnName = readData.columns.tolist()
columnName = columnName[1:]

podstawowe_statystyki = readData.describe()
print(podstawowe_statystyki)

numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()

correlation_matrix = np.corrcoef(numpy_array2, rowvar=False)
print(correlation_matrix)

numpy_array2 = ChangeVariablesToStimulants(numpy_array2,[1,7])




#Metoda Hellwiga
readData=pd.read_csv("DaneTelefonow.csv",sep=";")
numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()
numpy_array2 = ChangeVariablesToStimulants(numpy_array2,[1,6])

meanArray = getMeansArrayOfColumn(numpy_array2)
stdArray = getStandardDeviationArrayOfColumn(numpy_array2)

stddat = GetStandarizatedArray(numpy_array2,meanArray,stdArray)
optimalObject = getOptimalObject(stddat)
weightForPersonWhoWantCheapPhone = np.array([0.15,0.5,0.08,0.07,0.07,0.02,0.04,0.07])
weightForGamer = np.array([0.05,0.05,0.2,0.2,0.05,0.25,0.15,0.05])
weightForPhotographer = np.array([0.04,0.04,0.17,0.04,0.15,0.03,0.1,0.43])
weightForNormalPerson = np.array([0.3,0.14,0.07,0.07,0.2,0.05,0.07,0.1])
distance = DistanceFromObjectToMasterObject(stddat,optimalObject,weightForNormalPerson)


print(meanArray)
print(stdArray)
print(optimalObject)



print("Dystanse"+str(distance)+"")
print("Maksymalny dystans"+str(DistanceAsFarAsPossible(distance))+"")
finalResult=getFinalResult(distance,DistanceAsFarAsPossible(distance))
finalDataFrame = pd.DataFrame({'Telefony': readData.iloc[:,0], 'Wynik': finalResult})
print(finalDataFrame.sort_values(by='Wynik',ascending=False))




#Metoda Topsis

readData=pd.read_csv("DaneTelefonow.csv",sep=";")
numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()
numpy_array2 = ChangeVariablesToStimulants(numpy_array2,[1,6])

normalizedMatrix = normalizedMatrixTopsisMethod(numpy_array2)

theBestObject = getOptimalObject(normalizedMatrix)
theWorstObject = getTheWorstObject(normalizedMatrix)



#distance to the best object
diPlus = DistanceFromObjectToMasterObject(normalizedMatrix,theBestObject)
#distance to the worst object
diMinus = DistanceFromObjectToMasterObject(normalizedMatrix,theWorstObject)

Ri = getRiInTopsisMethod(diMinus,diPlus)


finalDataFrame = pd.DataFrame({'Telefony': readData.iloc[:,0], 'Wynik': Ri})
print(finalDataFrame.sort_values(by='Wynik'))








#ANALIZA Skupień


# Number of clusters (centers)
#readData=pd.read_csv("DaneTelefonowBezOutsiderow.csv",sep=";")
readData=pd.read_csv("DaneTelefonow.csv",sep=";")

print(readData)
xindexes = [i for i in range(1, 9) if i != 3]
objectIndexes = [i for i in range(readData.shape[0]) if i not in [8,9,11,12,20,22] ]

readData = readData.iloc[objectIndexes,xindexes]




numpy_array2 = readData.astype(float).to_numpy()

correlation_matrix = np.corrcoef(numpy_array2, rowvar=False)

sns.set(style="white")
columnName = readData.columns.tolist()



# Tworzenie wykresu macierzy korelacji
plt.figure(figsize=(10, 7))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, fmt=".2f", linewidths=.5)
heatmap.set_xticklabels(columnName, rotation=90)
heatmap.set_yticklabels(columnName, rotation = 0)

heatmap.xaxis.set_ticks_position('top')
heatmap.xaxis.set_label_position('top')
plt.title("Macierz korelacji parametrów")
plt.show()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(numpy_array2)


nClusters = 2

# Number of times the k-means algorithm will be run with different centroid seeds (nstart)
nstart = 10

# Create and fit the KMeans model
kmeansMethod = KMeans(n_clusters=nClusters, n_init=nstart, random_state=0).fit(scaled_data)

# Cluster centers
centers = kmeansMethod.cluster_centers_

labels = kmeansMethod.predict(scaled_data)



readData["grupa"] = labels


readData.columns = ['Ocena','Cena','RAM','Bateria','Z. procesora','r. kwartalow', 'mpx','grupa']


print(readData)
srednie = readData.groupby('grupa').mean()
print(srednie)

"""
plt.figure(figsize=(12, 10))

pair_plot = sns.pairplot(readData, hue='grupa', palette='bright')
for ax in pair_plot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), rotation = 90, labelpad  = 10)
    ax.set_ylabel(ax.get_ylabel(), rotation = 0, labelpad = 40)
pair_plot.fig.suptitle("Pair Plot of Phone Data with Cluster Membership", y=1.02)

plt.show()


fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(20, 5))

for i in range(6):
    ax = axes[i]
    ax.boxplot(numpy_array2[:, i])
    ax.set_title(f'{readData.columns[i]}')
    ax.set_ylabel('Wartości')
    ax.grid(True)



plt.tight_layout()
plt.show()






# =========== Metoda K-średnich =================

# Metoda lokcia

# Próba różnych wartości liczby klastrów (od 1 do 10) i obliczanie wartości odległości
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
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
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Obliczanie współczynnika silhouette dla każdego punktu
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

    # Obliczanie wartości współczynnika silhouette dla poszczególnych klastrów
    sample_silhouette_values = silhouette_samples(scaled_data, cluster_labels)

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
    kmedoids.fit(scaled_data)
    wcss.append(kmedoids.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Metoda łokcia (Elbow Method)')
plt.xlabel('Liczba klastrów')
plt.ylabel('Odległość wewnątrz klastrów (WCSS)')
plt.grid()
plt.show()




# Silhouette Method
n_clusters_range = range(2, 7)
silhouette_scores = []

for n_clusters in n_clusters_range:
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmedoids.fit_predict(scaled_data)

    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
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
    cluster_labels = kmedoids.fit_predict(scaled_data)

    # Calculate the average silhouette score
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg}")

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(scaled_data, cluster_labels)

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




# Parametry K-Medoids
nClusters = 5


# Tworzenie i dopasowanie modelu K-Medoids
kmedoids = KMedoids(n_clusters=nClusters, init='k-medoids++', random_state=0).fit(scaled_data)

# Centra klastrów (medoidy)
centers = kmedoids.cluster_centers_

# Przypisanie etykiet klastrów do danych
labels = kmedoids.predict(scaled_data)

# Dodanie etykiet klastrów do DataFrame
readData["grupa"] = labels

# Zmiana nazw kolumn
readData.columns = ['Ocena','Cena','RAM','Bateria','Z. procesora','r. kwartalow', 'mpx','grupa']

# Wyświetlenie danych
print(readData)

# Obliczenie średnich dla każdej grupy
srednie = readData.groupby('grupa').mean()

# Wyświetlenie średnich
print(srednie)


plt.figure(figsize=(12, 10))

pair_plot = sns.pairplot(readData, hue='grupa', palette='bright')
for ax in pair_plot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), rotation = 90, labelpad  = 10)
    ax.set_ylabel(ax.get_ylabel(), rotation = 0, labelpad = 40)
pair_plot.fig.suptitle("Pair Plot of Phone Data with Cluster Membership", y=1.02)

"""



import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram




scaler = StandardScaler()
data_scaled = scaler.fit_transform(numpy_array2)

# Wyznaczenie macierzy odległości
distance_matrix = ssd.pdist(data_scaled, 'euclidean')
print(distance_matrix)

condensed_matrix = ssd.squareform(distance_matrix)

# Grupowanie hierarchiczne
dataSet = sch.linkage(condensed_matrix, method='ward')


phone_names = readData.iloc[:,0]
print(phone_names)

plt.figure(figsize=(12, 10))
dendrogram(dataSet, color_threshold=1.5)  # This changes the color threshold
plt.show()



labels = [0,0]
labels[0] = sch.fcluster(dataSet, t=15, criterion='distance')
labels[1] = sch.fcluster(dataSet, t=8, criterion='distance')
print(labels)
silhouette_avg = [silhouette_score(data_scaled, labels[0]),silhouette_score(data_scaled, labels[1])]
davies_bouldin_index = [davies_bouldin_score(data_scaled, labels[0]),davies_bouldin_score(data_scaled, labels[1])]
calinski_harabasz_index = [calinski_harabasz_score(data_scaled, labels[0]),calinski_harabasz_score(data_scaled, labels[1])]

print("wynik Silhouette przy przycieciu 15:", silhouette_avg[0])
print("wynik Daviesa przy przycieciu 15", davies_bouldin_index[0])
print("wynik Calinski przy przycieciu 15", calinski_harabasz_index[0])


print("wynik Silhouette przy przycieciu 8:", silhouette_avg[1])
print("wynik Daviesa przy przycieciu 8", davies_bouldin_index[1])
print("wynik Calinski przy przycieciu 8", calinski_harabasz_index[1])


















