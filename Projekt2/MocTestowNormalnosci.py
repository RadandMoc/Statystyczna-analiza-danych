from enum import Enum
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import bartlett
from statsmodels.formula.api import ols
import random

class Normality_test(Enum):
    Shapiro_Wilk = "Shapiro-Wilk"
    Anderson_Darling = "Anderson-Darling"
    Lilliefors = "Lilliefors"
    Jarque_Bera = "Jarque-Bera"


class NormalDistribution:
    def __init__(self, n, mean, std_dev):
        self.n = n
        self.mean = mean
        self.std_dev = std_dev
        self.data = np.random.normal(self.mean, self.std_dev, self.n)

    def get_data(self):
        return self.data
    
class tStudent:
    def __init__(self, n, degrees_of_freedom):
        self.n = n
        self.degrees_of_freedom = degrees_of_freedom
        self.data = np.random.standard_t(degrees_of_freedom, n)

    def get_data(self):
        return self.data


class gammaFunction:
   def __init__(self, n, shape, scale):
        self.n = n
        self.shape = shape
        self.scale = scale
        self.data = np.random.gamma(self.shape, self.scale, self.n)



class logNormal:
    def __init__(self, n, mean, sigma):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.data = np.random.lognormal(self.mean, self.sigma, self.n)

    def get_data(self):
        return self.data



def check_test_power(normality_test,data):
    if normality_test == Normality_test.Shapiro_Wilk:
        statistic, p_value = stats.shapiro(data)
        return p_value #< 0.05
    elif normality_test == Normality_test.Anderson_Darling:
        result = stats.anderson(data)
        return result.statistic #> result.critical_values[2]
    elif normality_test == Normality_test.Lilliefors:
        statistic, p_value = sm.stats.diagnostic.lilliefors(data)
        return p_value #< 0.05
    elif normality_test == Normality_test.Jarque_Bera:
        statistic, p_value = stats.jarque_bera(data)
        return p_value #< 0.05
    else:
        raise Exception("Error in input variable normality_test")
        



def normal_distribution_power_test(list_number_of_datas, list_of_std, ifNormal,  iterations = 10000):
   
    results = []

    for n in list_number_of_datas:
        for sd in list_of_std:
            ts = []
            if ifNormal == True:
                ts = [NormalDistribution(n, 0, sd).data for _ in range(iterations)]
            else:
                ts = [logNormal(n,0,sd).data for _ in range(iterations)]
            power_shapiro_wilk = (sum(check_test_power(Normality_test.Shapiro_Wilk, data) for data in ts) / iterations)
            power_anderson_darling = (sum(check_test_power(Normality_test.Anderson_Darling, data) for data in ts) / iterations)
            power_lilliefors = (sum(check_test_power(Normality_test.Lilliefors, data) for data in ts) / iterations)
            power_jarque_bera = (sum(check_test_power(Normality_test.Jarque_Bera, data) for data in ts) / iterations)
            results.append([n, sd, power_shapiro_wilk, power_anderson_darling, power_lilliefors, power_jarque_bera])
    
    df = pd.DataFrame(results, columns=['Liczba danych', 'Odchylenie', 'Shapiro-Wilk', 'Anderson-Darling', 'Lilliefors','Jarque-Bera'])
    return df

def t_student_distribution_power_test(list_number_of_datas, list_of_degrees_of_freedom, iterations = 10000):
    results = []

    for n in list_number_of_datas:
        for sd in list_of_degrees_of_freedom:
            ts = [tStudent(n,sd).data for _ in range(iterations)]
            power_shapiro_wilk = (sum(check_test_power(Normality_test.Shapiro_Wilk, data) for data in ts) / iterations)
            power_anderson_darling = (sum(check_test_power(Normality_test.Anderson_Darling, data) for data in ts) / iterations)
            power_lilliefors = (sum(check_test_power(Normality_test.Lilliefors, data) for data in ts) / iterations)
            power_jarque_bera = (sum(check_test_power(Normality_test.Jarque_Bera, data) for data in ts) / iterations)
            results.append([n, sd, power_shapiro_wilk, power_anderson_darling, power_lilliefors, power_jarque_bera])
    df = pd.DataFrame(results, columns=['Liczba danych', 'Stopnie swobody', 'Shapiro-Wilk', 'Anderson-Darling', 'Lilliefors','Jarque-Bera'])
    return df


def gamma_distribution_power_test(list_number_of_datas,shape,scale,iterations = 10000):
    results = []
    for n in list_number_of_datas:
        for sh in shape:
            for sc in scale:
                ts = [gammaFunction(n,sh,sc).data for _ in range(iterations)]
                power_shapiro_wilk = sum(check_test_power(Normality_test.Shapiro_Wilk, data) for data in ts) / iterations
                power_anderson_darling = sum(check_test_power(Normality_test.Anderson_Darling, data) for data in ts) / iterations
                power_lilliefors = sum(check_test_power(Normality_test.Lilliefors, data) for data in ts) / iterations
                power_jarque_bera = sum(check_test_power(Normality_test.Jarque_Bera, data) for data in ts) / iterations
                results.append([n,sh, sc, power_shapiro_wilk, power_anderson_darling, power_lilliefors, power_jarque_bera])
    df = pd.DataFrame(results, columns=['Liczba danych', 'Ksztalt' ,'Skala', 'Shapiro-Wilk', 'Anderson-Darling', 'Lilliefors','Jarque-Bera'])
    return df


def plot_test_powers_by_sample_size(df,text):
    std_devs = df[text].unique()
    tests = df.columns[2:]
    for sd in std_devs:
        plt.figure(figsize=(12, 8))
        for test in tests:
            subset = df[df[text] == sd]
            plt.plot(subset['Liczba danych'], subset[test], label=test)
        plt.xlabel('Liczba danych w próbce')
        plt.ylabel('Moc testu gdy dane pochodzą z rozkładu lognormalnego')
        plt.xticks(subset['Liczba danych'])  
        plt.title(f'Moc różnych testów normalności dla {text} = {sd}')
        plt.legend()
        plt.show()




def plot_gamma_test_powers_combined(df):
    shape_values = df['Ksztalt'].unique()
    scale_values = df['Skala'].unique()
    tests = df.columns[3:]  

    for shape in shape_values:
        for scale in scale_values:
            subset = df[(df['Ksztalt'] == shape) & (df['Skala'] == scale)]
            if not subset.empty:
                plt.figure(figsize=(12, 8))

                for test in tests:
                    plt.plot(subset['Liczba danych'], subset[test], label=test, marker='o')

                plt.xlabel('Liczba danych')
                plt.ylabel('Moc testu')
                plt.xticks(subset['Liczba danych'])  # Zapewnia, że osi x odpowiadają liczbie danych
                plt.title(f'Moc różnych testów normalności dla kształtu = {shape} i skali = {scale}')
                plt.legend()
                plt.grid(True)
                plt.show()


def cut_data_to_the_same_size(data):
    numbers_of_datas = list(range(7))
    matrix_of_colors = ["Black","White","Blue","Silver","Red","Green","Gold"]
    for i in range(7):
            numbers_of_datas[i] = np.count_nonzero(data[4] == matrix_of_colors[i])
    how_big_groups = min(numbers_of_datas)
    for i in range(7):
        indeksy = np.where(data[:, 4] == matrix_of_colors[i])
        wanted_data = data[indeksy]
        if how_big_groups < np.shape(wanted_data)[0]:
            indices_for_train =  random.sample(range(0, wanted_data.shape[0]), int(how_big_groups))
            new_data = wanted_data[indices_for_train,:]
        else:
            new_data = wanted_data
        if i == 0:
            returner = new_data
        else:
            np.vstack([returner,new_data])
    return returner

number_of_data = [10] + list(range(25,1000,25))

std = [1,3,5,10,20,50]
logstd = [1/32,1/16,1/8,1/4,1/2,1,3/2]
list_of_degrees_of_freedom = [2,5,10,50,100,500,1000,5000]
shape = [3,3.5,4,4.5]
scale = [0.35,0.5,0.65,0.8]



#Funkcja plot_test_powers_by_sample_size miała zmieniane tytuły wykresów w zależności od tego jaki był rozkład
#poniższy kod jest zakomentowany gdyż działa on za długo aby go w rozsadnym czasie przetestewać. Aby go przetestować można zmienić wartość iterations.
"""
normal = normal_distribution_power_test(number_of_data, std, True,10)
print(plot_test_powers_by_sample_size(normal,"Odchylenie"))

t_Student = t_student_distribution_power_test(number_of_data, list_of_degrees_of_freedom)
print(plot_test_powers_by_sample_size(t_Student,"Stopnie swobody"))

lognormal = normal_distribution_power_test(number_of_data, logstd, False)
print(plot_test_powers_by_sample_size(lognormal,"Odchylenie"))


gamma_results = gamma_distribution_power_test(number_of_data,shape,scale)
plot_gamma_test_powers_combined(gamma_results)

"""



#ANOVA
DATA = pd.read_csv("DaneTelefonow.csv",sep=";", decimal=",")
black = DATA[DATA['Kolor'] == "Czarny"]["Cena"]
blue = DATA[DATA['Kolor'] == "Niebieski"]["Cena"]
white = DATA[DATA['Kolor'] == "Bialy"]["Cena"]
silver = DATA[DATA['Kolor'] == "Srebrny"]["Cena"]
yellow = DATA[DATA['Kolor'] == "Zolty"]["Cena"]
green = DATA[DATA['Kolor'] == "Zielony"]["Cena"]
red = DATA[DATA['Kolor'] == "Czerwony"]["Cena"]


#Średnie ceny
#print(sum(DATA['Cena']/len(DATA)))

#Średnie w grupie
#print(sum(yellow)/len(yellow))
#print(sum(white)/len(white))
#print(sum(blue)/len(blue))
#print(sum(green)/len(green))
#print(sum(silver)/len(silver))
#print(sum(black)/len(black))
#print(sum(red)/len(red))

#liczebność danych
#print(len(yellow))
#print(len(white))
#print(len(blue))
#print(len(green))
#print(len(silver))
#print(len(black))
#print(len(red))

print("Rezultaty testów Shapir-Wilka:")
print(check_test_power(Normality_test.Shapiro_Wilk,black))
print(check_test_power(Normality_test.Shapiro_Wilk,blue))
print(check_test_power(Normality_test.Shapiro_Wilk,white))
print(check_test_power(Normality_test.Shapiro_Wilk,silver))
print(check_test_power(Normality_test.Shapiro_Wilk,yellow))
print(check_test_power(Normality_test.Shapiro_Wilk,red))
print(check_test_power(Normality_test.Shapiro_Wilk,green))



stat, p = bartlett(black, blue, white, silver, yellow)
print(f"Nie ma podstaw do odrzucenia hipotezy zerowej p-value >0.05 {p>0.05} wartosc wynosi {p}")


model = ols('Cena ~ Kolor', data=DATA).fit()
anova_result = sm.stats.anova_lm(model, typ=1)
print("Oto rezultat testu ANOVY:")
print(anova_result)



Apple = DATA[DATA['Marka'] == "Apple"]["Cena"]
Samsung = DATA[DATA['Marka'] == "Samsung"]["Cena"]
Huawei = DATA[DATA['Marka'] == "Huawei"]["Cena"]
Xiaomi = DATA[DATA['Marka'] == "Xiaomi"]["Cena"]

#Liczności w grupach marek:
#print(len(Apple))
#print(len(Samsung))
#print(len(Huawei))
#print(len(Xiaomi))


print("Oto sprawdzenia normalności w grupach:")
print(check_test_power(Normality_test.Shapiro_Wilk,Apple))
print(check_test_power(Normality_test.Shapiro_Wilk,Samsung))
print(check_test_power(Normality_test.Shapiro_Wilk,Huawei))
print(check_test_power(Normality_test.Shapiro_Wilk,Xiaomi))




stat, p = bartlett(Apple, Samsung, Huawei, Xiaomi)
print(f"Nie ma podstaw do odrzucenia hipotezy zerowej p-value >0.05 {p>0.05} wartosc wynosi {p}")


statystyka, p_value = stats.kruskal(Apple, Samsung, Huawei, Xiaomi)
print(p_value)
if p_value < 0.05:
    print("Istnieja istotne statystycznie roznice miedzy grupami.")
else:
    print("Nie ma istotnych statystycznie roznic miedzy grupami.")


