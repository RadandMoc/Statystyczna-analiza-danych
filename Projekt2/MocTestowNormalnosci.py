from enum import Enum
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import bartlett
from scipy.stats import levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import f_oneway
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
        return p_value < 0.05
    elif normality_test == Normality_test.Anderson_Darling:
        result = stats.anderson(data)
        return result.statistic > result.critical_values[2]
    elif normality_test == Normality_test.Lilliefors:
        statistic, p_value = sm.stats.diagnostic.lilliefors(data)
        return p_value < 0.05
    elif normality_test == Normality_test.Jarque_Bera:
        statistic, p_value = stats.jarque_bera(data)
        return p_value < 0.05
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

            print(f"Moc testu Shapiro-Wilka dla {n} danych o odchyleniu {sd}:", power_shapiro_wilk)
            print(f"Moc testu Anderson_Darling dla {n} danych o odchyleniu {sd}:", power_anderson_darling)
            print(f"Moc testu Lilliefors dla {n} danych o odchyleniu {sd}", power_lilliefors)
            print(f"Moc testu Jarque_Bera dla {n} danych o odchyleniu {sd}", power_jarque_bera)
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

            print(f"Moc testu Shapiro-Wilka dla {n} danych o odchyleniu {sd}:", power_shapiro_wilk)
            print(f"Moc testu Anderson_Darling dla {n} danych o odchyleniu {sd}:", power_anderson_darling)
            print(f"Moc testu Lilliefors dla {n} danych o odchyleniu {sd}", power_lilliefors)
            print(f"Moc testu Jarque_Bera dla {n} danych o odchyleniu {sd}", power_jarque_bera)
            results.append([n, sd, power_shapiro_wilk, power_anderson_darling, power_lilliefors, power_jarque_bera])
    df = pd.DataFrame(results, columns=['Liczba danych', 'Stopnie swobody', 'Shapiro-Wilk', 'Anderson-Darling', 'Lilliefors','Jarque-Bera'])
    return df


def gamma_distribution_power_test(list_number_of_datas,shape,scale,iterations = 10000):
    for n in list_number_of_datas:
        for sh in shape:
            for sc in scale:
                ts = [gammaFunction(n,sh,sc).data for _ in range(iterations)]
                power_shapiro_wilk = 1 - (sum(check_test_power(Normality_test.Shapiro_Wilk, data) for data in ts) / iterations)
                power_anderson_darling = 1- (sum(check_test_power(Normality_test.Anderson_Darling, data) for data in ts) / iterations)
                power_lilliefors = 1-(sum(check_test_power(Normality_test.Lilliefors, data) for data in ts) / iterations)
                power_jarque_bera = 1-(sum(check_test_power(Normality_test.Jarque_Bera, data) for data in ts) / iterations)

                print(f"Moc testu Shapiro-Wilka dla {n} danych o shape {sh}:", power_shapiro_wilk)
                print(f"Moc testu Anderson_Darling dla {n} danych o shape {sh}:", power_anderson_darling)
                print(f"Moc testu Lilliefors dla {n} danych o shape {sh}", power_lilliefors)
                print(f"Moc testu Jarque_Bera dla {n} danych o shape {sh}", power_jarque_bera)



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
        plt.xticks(subset['Liczba danych'])  # Ensure x-ticks represent sample sizes

        plt.title(f'Moc różnych testów normalności dla {text} = {sd}')
        plt.legend()
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
print(number_of_data)
std = [1,3,5,10,20,50]
logstd = [1/16,1/8,1/4,1/2,1,3/2,5,10]
list_of_degrees_of_freedom = [2,5,10,50,100,500,1000,5000]
shape = [1,2,5,10,20]
scale = [1,2,4,8,12]


#normal = normal_distribution_power_test(number_of_data, std, True)

#print(plot_test_powers_by_sample_size(normal,"Odchylenie"))

#t_Student = t_student_distribution_power_test(number_of_data, list_of_degrees_of_freedom,iterations=10000)
#print(t_Student.iloc[:,:3])
#print(t_Student.iloc[:,3:])


#print(plot_test_powers_by_sample_size(t_Student,"Stopnie swobody"))

"""

lognormal = normal_distribution_power_test(number_of_data, logstd, False,iterations=100)
print(plot_test_powers_by_sample_size(lognormal,"Odchylenie"))
"""



#gamma_distribution_power_test(number_of_data,shape,scale)







#ANOVA
DATA = pd.read_csv("DaneMalaIloscKolorow.csv",sep=";", decimal=",")
#dane = DATA[DATA['Telefon\zmienne'].str.startswith("Apple")]
#daneSAMSUNG = DATA[DATA['Telefon\zmienne'].str.startswith("SAMSUNG")]["Cena"]
#daneHuawei = DATA[DATA['Marka'] == "Huawei"]["Cena"]
#Czarny = DATA[DATA['Kolor'] == "Lawendowy"]["Cena"]

#print(check_test_power(Normality_test.Jarque_Bera,dane["Cena"]))
#print(check_test_power(Normality_test.Jarque_Bera,daneSAMSUNG))
#print(check_test_power(Normality_test.Jarque_Bera,daneHuawei))


#print(check_test_power(Normality_test.Jarque_Bera,Czarny))

#stat, p = bartlett(dane, daneSAMSUNG, daneHuawei)
#print(f"Nie ma podstaw do odrzucenia hipotezy zerowej p-value >0.05 {p>0.05} wartosc wynosi {p}")


#stat, p = levene(dane, daneSAMSUNG, daneHuawei)

#print("Statystyka testu:", stat)
#print("P-wartość:", p)



#print(data_renamed[~data_renamed["General_Color"].isin(["Black", "White", "Red", "Green", "Silver", "Golden", "Blue"])]["General_Color"].unique())
#print(len(data_renamed[~data_renamed["General_Color"].isin(["Black", "White", "Red", "Green", "Silver", "Golden", "Blue"])]["General_Color"].unique()))

"""
model = ols('Selling_Price ~ Color', data=data_renamed).fit()
anova_result = sm.stats.anova_lm(model, typ=1)
print(anova_result)
"""