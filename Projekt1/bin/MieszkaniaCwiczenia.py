import pandas as pd
import numpy as np
def ChangeCommaToPoint(text):
    df_skopiowany = text.copy()  # Tworzymy kopię dataframe, aby nie zmieniać oryginalnego obiektu
    
    # Iterujemy po każdej komórce DataFrame i zamieniamy przecinki na kropki
    for kolumna in df_skopiowany.columns:
        if df_skopiowany[kolumna].dtype == 'object':  # Sprawdzamy tylko kolumny zawierające tekst
            df_skopiowany[kolumna] = df_skopiowany[kolumna].astype(str).str.replace(',', '.')
    
    return df_skopiowany

def printBasicStats(data):
    pd.set_option('display.max_columns', None)  # Wyświetlanie wszystkich kolumn
    pd.set_option('display.max_rows', None) 
    columnName = readData.columns.tolist()
    columnName = columnName[1:]
    podstawowe_statystyki = data.describe()
    print(podstawowe_statystyki)

def calculate_SS_effect(dataframe, column_name, target_column):
    # Grupowanie danych ze względu na kolumnę 'column_name' i obliczanie średnich
    group_means = dataframe.groupby(column_name)[target_column].mean()
    
    # Obliczanie średniej ogólnej
    overall_mean = dataframe[target_column].mean()
    
    # Inicjalizacja sumy kwadratów efektów
    SS_effect = 0
    
    # Obliczanie SS_effect dla każdej grupy
    for group, mean in group_means.items():
        n = len(dataframe[dataframe[column_name] == group])
        SS_effect += n * (mean - overall_mean) ** 2
    
    return SS_effect

def calculate_SS_error(dataframe, column_name, target_column):
    # Obliczanie całkowitej liczby obserwacji
    total_obs = len(dataframe)
    
    # Obliczanie liczby unikalnych grup
    unique_groups = dataframe[column_name].unique()
    num_groups = len(unique_groups)
    
    # Inicjalizacja sumy kwadratów błędów
    SS_error = 0
    
    # Obliczanie SS_error dla każdej grupy
    for group in unique_groups:
        group_data = dataframe[dataframe[column_name] == group][target_column]
        group_mean = group_data.mean()
        n = len(group_data)
        SS_error += np.sum((group_data - group_mean) ** 2)
    
    return SS_error

def calculate_SS_total(dataframe, target_column):
    # Obliczanie średniej ogólnej
    overall_mean = dataframe[target_column].mean()
    
    # Obliczanie SStotal
    SS_total = np.sum((dataframe[target_column] - overall_mean) ** 2)
    
    return SS_total

readData = pd.read_csv("mieszkania.csv",sep=";")
readData = ChangeCommaToPoint(readData)


readData.iloc[:,0:1] = readData.iloc[:,0:1].astype(float)

printBasicStats(readData)
print(readData)
dataCopy = readData
SS_effect_result_dzielnice = calculate_SS_effect(dataCopy, 'dzielnica', 'cena')
print("Suma kwadratów dzielnic efektów (SS_effect):", SS_effect_result_dzielnice)
SS_effect_result_typ_budynku = calculate_SS_effect(dataCopy, 'typ budynku', 'cena')
print("Suma kwadratów typu budynku efektów (SS_effect):", SS_effect_result_typ_budynku)
SS_error_result = calculate_SS_error(dataCopy, 'dzielnica', 'cena')
print("Suma kwadratów błędów dzielnic (SS_error):", SS_error_result)
SS_error_result = calculate_SS_error(dataCopy, 'typ budynku', 'cena')
print("Suma kwadratów błędów typ budynku (SS_error):", SS_error_result)
SStotal_result = calculate_SS_total(dataCopy, 'cena')
print("Suma kwadratów całkowitych (SStotal):", SStotal_result)
etaKwadrat = (SS_effect_result_dzielnice/SStotal_result)*100
print("Wynik eta^2:", etaKwadrat,"%")