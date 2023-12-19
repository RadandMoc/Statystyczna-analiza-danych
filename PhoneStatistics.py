import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

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

def ChangeCommaToPoint(text):
    df_skopiowany = text.copy()  # Tworzymy kopię dataframe, aby nie zmieniać oryginalnego obiektu
    
    # Iterujemy po każdej komórce DataFrame i zamieniamy przecinki na kropki
    for kolumna in df_skopiowany.columns:
        if df_skopiowany[kolumna].dtype == 'object':  # Sprawdzamy tylko kolumny zawierające tekst
            df_skopiowany[kolumna] = df_skopiowany[kolumna].astype(str).str.replace(',', '.')
    
    return df_skopiowany

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
    
def printBasicStats(data):
    pd.set_option('display.max_columns', None)  # Wyświetlanie wszystkich kolumn
    pd.set_option('display.max_rows', None) 
    columnName = readData.columns.tolist()
    columnName = columnName[1:]
    podstawowe_statystyki = data.describe()
    print(podstawowe_statystyki)
    



def HellwigMethod(readData,indexesForVariablesToStimulants, weight = None):
    
    # Process the data
    numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()
    numpy_array2 = ChangeVariablesToStimulants(numpy_array2, indexesForVariablesToStimulants)

    meanArray = getMeansArrayOfColumn(numpy_array2)
    stdArray = getStandardDeviationArrayOfColumn(numpy_array2)

    stddat = GetStandarizatedArray(numpy_array2, meanArray, stdArray)
    optimalObject = getOptimalObject(stddat)
    
    # Calculate distances
    distance = DistanceFromObjectToMasterObject(stddat, optimalObject, weight)
    max_distance = DistanceAsFarAsPossible(distance)
    finalResult = getFinalResult(distance, max_distance)

    # Create final DataFrame
    finalDataFrame = pd.DataFrame({'Telefony': readData.iloc[:,0], 'Wynik': finalResult})
    return finalDataFrame.sort_values(by='Wynik', ascending=False)




def displayHellwigResults(readData, indexes, consumerType, weight = None):
    results = HellwigMethod(readData, indexes, weight)
    sortedResults = results.sort_values(by='Wynik', ascending=False)

    print(f"Hellwig Dla {consumerType}")
    print(sortedResults)



def TOPSIS_Method(readData, indexesForVariablesToStimulants,weight = None):

    dataMatrix = readData.iloc[:, 1:].astype(float).to_numpy()

    dataMatrix = ChangeVariablesToStimulants(numpy_array2, indexesForVariablesToStimulants)

    normalizedMatrix = normalizedMatrixTopsisMethod(dataMatrix)

    theBestObject = getOptimalObject(normalizedMatrix)
    theWorstObject = getTheWorstObject(normalizedMatrix)


    diPlus = DistanceFromObjectToMasterObject(normalizedMatrix, theBestObject,weight)
    diMinus = DistanceFromObjectToMasterObject(normalizedMatrix, theWorstObject,weight)

    Ri = getRiInTopsisMethod(diMinus, diPlus)

 
    finalDataFrame = pd.DataFrame({'Telefony': readData.iloc[:, 0], 'Wynik': Ri})
    return finalDataFrame.sort_values(by='Wynik')

def displayTopsisMethod(readData, indexesForVariablesToStimulants, consumerType,weight = None):
    sortedResults = TOPSIS_Method(readData, indexesForVariablesToStimulants, weight)
    print(f"Topsis Dla {consumerType}")
    print(sortedResults)


def elbow_method(scaled_data, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Metoda łokcia (Elbow Method)')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Odległość wewnątrz klastrów (WCSS)')
    plt.grid()
    plt.show()
    return wcss


def calculate_silhouette_scores(scaled_data, n_clusters_range, show_plots=True):
    silhouette_scores = []

    for n_clusters in n_clusters_range:
        # Tworzenie instancji k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(scaled_data)

        # Obliczanie współczynnika silhouette
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        if show_plots:
            # Obliczanie wartości współczynnika silhouette dla poszczególnych klastrów
            sample_silhouette_values = silhouette_samples(scaled_data, cluster_labels)

            # Wykres dla każdego klastra
            y_lower = 10
            for i in range(n_clusters):
                # Zebranie wartości współczynnika silhouette dla punktów
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
    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.plot(n_clusters_range, silhouette_scores, marker='o')
        plt.title('Metoda profilu (Silhouette Method)')
        plt.xlabel('Liczba klastrów')
        plt.ylabel('Średni współczynnik silhouette')
        plt.grid()
        plt.show()

    return silhouette_scores


def plot_silhouette_scores(n_clusters_range, silhouette_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(n_clusters_range, silhouette_scores, marker='o')
    plt.title('Metoda profilu (Silhouette Method)')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Średni współczynnik silhouette')
    plt.grid()
    plt.show()

def create_pair_plot(data, hue, palette='bright', figsize=(12, 10), suptitle="Pair Plot with Cluster Membership"):
    plt.figure(figsize=figsize)

    pair_plot = sns.pairplot(data, hue=hue, palette=palette)
    for ax in pair_plot.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), rotation=90, labelpad=10)
        ax.set_ylabel(ax.get_ylabel(), rotation=0, labelpad=40)
    pair_plot.fig.suptitle(suptitle, y=1.02)

    plt.show()





def perform_kmeans_and_calculate_means(data, scaled_data, n_clusters, n_init=10, random_state=0):
    # Create and fit the KMeans model
    kmeans_model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit(scaled_data)

    # Predict the cluster for each data point
    labels = kmeans_model.predict(scaled_data)

    # Add the cluster labels to the original data
    data_with_clusters = data.copy()
    data_with_clusters["grupa"] = labels

    # Rename columns as per your specific dataset
    data_with_clusters.columns = ['Ocena', 'Cena', 'RAM', 'Bateria', 'Z. procesora', 'r. kwartalow', 'mpx', 'grupa']

    # Calculate the mean of each cluster
    cluster_means = data_with_clusters.groupby('grupa').mean()

    return data_with_clusters, cluster_means








def plot_correlation_matrix(correlation_matrix, column_names, figsize=(10, 7), cmap='coolwarm', fmt=".2f", linewidths=0.5, title="Macierz korelacji parametrów"):
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap=cmap, square=True, fmt=fmt, linewidths=linewidths)
    heatmap.set_xticklabels(column_names, rotation=90)
    heatmap.set_yticklabels(column_names, rotation=0)

    heatmap.xaxis.set_ticks_position('top')
    heatmap.xaxis.set_label_position('top')
    plt.title(title)
    plt.show()

def scale_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)



def perform_hierarchical_clustering(scaled_data, method='ward', metric='euclidean', figsize=(12, 10), color_threshold=1.5):
    distance_matrix = ssd.pdist(scaled_data, metric)
    condensed_matrix = ssd.squareform(distance_matrix)
    linkage_matrix = sch.linkage(condensed_matrix, method=method)

    plt.figure(figsize=figsize)
    sch.dendrogram(linkage_matrix, color_threshold=color_threshold)
    plt.show()

    return linkage_matrix

def calculate_cluster_scores(scaled_data, linkage_matrix, t_values=[8,15]):
    labels = [sch.fcluster(linkage_matrix, t=t, criterion='distance') for t in t_values]
    silhouette_scores = [silhouette_score(scaled_data, label) for label in labels]
    davies_bouldin_indexes = [davies_bouldin_score(scaled_data, label) for label in labels]
    calinski_harabasz_indexes = [calinski_harabasz_score(scaled_data, label) for label in labels]

    return {
        "labels": labels,
        "silhouette": silhouette_scores,
        "davies_bouldin": davies_bouldin_indexes,
        "calinski_harabasz": calinski_harabasz_indexes
    }



readData=pd.read_csv("DaneTelefonow.csv",sep=";")
readData = ChangeCommaToPoint(readData)

printBasicStats(readData)

numpy_array2 = readData.iloc[:,1:].astype(float).to_numpy()

#boxplot aby wyeliminować outlierow w metodzie skupien

fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))

for i in range(6):
    ax = axes[i]
    ax.boxplot(numpy_array2[:, i])
    ax.set_title(f'{readData.columns[i]}')
    ax.set_ylabel('Wartości')
    ax.grid(True)



plt.tight_layout()
plt.show()




correlation_matrix = np.corrcoef(numpy_array2, rowvar=False)
print(correlation_matrix)



#Metoda Hellwiga
readData=pd.read_csv("DaneTelefonow.csv",sep=";")


weightForPersonWhoWantCheapPhone = np.array([0.15,0.5,0.08,0.07,0.07,0.02,0.04,0.07])
weightForGamer = np.array([0.05,0.05,0.2,0.2,0.05,0.25,0.15,0.05])
weightForPhotographer = np.array([0.04,0.04,0.17,0.04,0.15,0.03,0.1,0.43])
weightForNormalPerson = np.array([0.3,0.14,0.07,0.07,0.2,0.05,0.07,0.1])


displayHellwigResults(readData, [1, 6], "bez wag")
displayHellwigResults(readData, [1, 6], "oszczednego konsumenta", weightForPersonWhoWantCheapPhone)
displayHellwigResults(readData, [1, 6], "gracza", weightForGamer)
displayHellwigResults(readData, [1, 6], "fotografa", weightForPhotographer)
displayHellwigResults(readData, [1, 6], "zwyklego uzytkownika", weightForNormalPerson)

#Metoda Topsis

displayTopsisMethod(readData, [1, 6], "bez wag")
displayTopsisMethod(readData, [1, 6], "oszczednego konsumenta", weightForPersonWhoWantCheapPhone)
displayTopsisMethod(readData, [1, 6], "gracza", weightForGamer)
displayTopsisMethod(readData, [1, 6], "fotografa", weightForPhotographer)
displayTopsisMethod(readData, [1, 6], "zwyklego uzytkownika", weightForNormalPerson)










#ANALIZA Skupień












# =========== Metoda K-średnich =================

#readData=pd.read_csv("DaneTelefonowBezOutsiderow.csv",sep=";")
readData=pd.read_csv("DaneTelefonow.csv",sep=";")

print(readData)
xindexes = [i for i in range(1, 9) if i != 3]
objectIndexes = [i for i in range(readData.shape[0]) if i not in [8,9,11,12,20,22] ]

readData = readData.iloc[objectIndexes,xindexes]

print(f"oto moje dane {readData}")


numpy_array2 = readData.astype(float).to_numpy()

correlation_matrix = np.corrcoef(numpy_array2, rowvar=False)

sns.set(style="white")
columnName = readData.columns.tolist()
plot_correlation_matrix(correlation_matrix,columnName)


# Metoda lokcia
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numpy_array2)

elbow_method(scaled_data, max_clusters=10)

#Metoda Profilu
n_clusters_range=range(2, 7)
silhouette_scores = calculate_silhouette_scores(scaled_data,n_clusters_range)
plot_silhouette_scores(n_clusters_range,silhouette_scores)



nClusters = 2 #po analizie
readData, srednie = perform_kmeans_and_calculate_means(readData, scaled_data, nClusters)

create_pair_plot(readData,'grupa')



# ============== Metoda K-Medoid =======================


wcss = []
for i in range(1, 11):
    kmedoids = KMedoids(n_clusters=i, init='k-medoids++', max_iter=300, random_state=0)
    kmedoids.fit(scaled_data)
    wcss.append(kmedoids.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Metoda łokcia (Elbow Method)')
plt.xlabel('Liczba klastrów')
plt.ylabel('Odległość wewnątrz klastrów (WCSS)')
plt.grid()
plt.show()





n_clusters_range = range(2, 7)
silhouette_scores = []

for n_clusters in n_clusters_range:
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmedoids.fit_predict(scaled_data)

    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 6))
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title('Metoda profilu (Silhouette Method) dla metody K-Medoids')
plt.xlabel('Numer klastra')
plt.ylabel('Średni współczynnik silhouette')
plt.grid()
plt.show()






n_clusters_range = range(2, 7)

for n_clusters in n_clusters_range:
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmedoids.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette score is : {silhouette_avg}")
    sample_silhouette_values = silhouette_samples(scaled_data, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10 

    plt.title(f"Metoda profilu (Silhouette Method) dla różnych klastrów = {n_clusters}")
    plt.xlabel("Średni współczynnik silhouette")
    plt.ylabel("Liczba klastrów")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  
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





import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram



scaled_data = scale_data(numpy_array2)

# Step 2: Perform hierarchical clustering and plot dendrogram
linkage_matrix = perform_hierarchical_clustering(scaled_data)

# Step 3: Calculate cluster scores
scores = calculate_cluster_scores(scaled_data, linkage_matrix, t_values=[15, 8])

print(scores)
