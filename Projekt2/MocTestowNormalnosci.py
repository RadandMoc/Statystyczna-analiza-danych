from enum import Enum
import numpy as np
import statsmodels.api as sm
from scipy import stats


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
        



def normal_distribution_power_test(list_number_of_datas, list_of_std, ifNormal):
    iterations = 1000
    for n in list_number_of_datas:
        for sd in list_of_std:
            ts = []
            if ifNormal == True:
                ts = [NormalDistribution(n, 0, sd).data for _ in range(iterations)]
            else:
                ts = [logNormal(n,0,sd).data for _ in range(iterations)]
            power_shapiro_wilk = 1 - (sum(check_test_power(Normality_test.Shapiro_Wilk, data) for data in ts) / iterations)
            power_anderson_darling = 1- (sum(check_test_power(Normality_test.Anderson_Darling, data) for data in ts) / iterations)
            power_lilliefors = 1-(sum(check_test_power(Normality_test.Lilliefors, data) for data in ts) / iterations)
            power_jarque_bera = 1-(sum(check_test_power(Normality_test.Jarque_Bera, data) for data in ts) / iterations)

            print(f"Moc testu Shapiro-Wilka dla {n} danych o odchyleniu {sd}:", power_shapiro_wilk)
            print(f"Moc testu Anderson_Darling dla {n} danych o odchyleniu {sd}:", power_anderson_darling)
            print(f"Moc testu Lilliefors dla {n} danych o odchyleniu {sd}", power_lilliefors)
            print(f"Moc testu Jarque_Bera dla {n} danych o odchyleniu {sd}", power_jarque_bera)


def t_student_distribution_power_test(list_number_of_datas, list_of_degrees_of_freedom):
    iterations = 1000
    for n in list_number_of_datas:
        for sd in list_of_degrees_of_freedom:
            ts = [tStudent(n,sd).data for _ in range(iterations)]
            power_shapiro_wilk = 1 - (sum(check_test_power(Normality_test.Shapiro_Wilk, data) for data in ts) / iterations)
            power_anderson_darling = 1- (sum(check_test_power(Normality_test.Anderson_Darling, data) for data in ts) / iterations)
            power_lilliefors = 1-(sum(check_test_power(Normality_test.Lilliefors, data) for data in ts) / iterations)
            power_jarque_bera = 1-(sum(check_test_power(Normality_test.Jarque_Bera, data) for data in ts) / iterations)

            print(f"Moc testu Shapiro-Wilka dla {n} danych o odchyleniu {sd}:", power_shapiro_wilk)
            print(f"Moc testu Anderson_Darling dla {n} danych o odchyleniu {sd}:", power_anderson_darling)
            print(f"Moc testu Lilliefors dla {n} danych o odchyleniu {sd}", power_lilliefors)
            print(f"Moc testu Jarque_Bera dla {n} danych o odchyleniu {sd}", power_jarque_bera)


def t_student_distribution_power_test(list_number_of_datas, list_of_degrees_of_freedom):
    iterations = 1000
    for n in list_number_of_datas:
        for sd in list_of_degrees_of_freedom:
            ts = [tStudent(n,sd).data for _ in range(iterations)]
            power_shapiro_wilk = 1 - (sum(check_test_power(Normality_test.Shapiro_Wilk, data) for data in ts) / iterations)
            power_anderson_darling = 1- (sum(check_test_power(Normality_test.Anderson_Darling, data) for data in ts) / iterations)
            power_lilliefors = 1-(sum(check_test_power(Normality_test.Lilliefors, data) for data in ts) / iterations)
            power_jarque_bera = 1-(sum(check_test_power(Normality_test.Jarque_Bera, data) for data in ts) / iterations)

            print(f"Moc testu Shapiro-Wilka dla {n} danych o odchyleniu {sd}:", power_shapiro_wilk)
            print(f"Moc testu Anderson_Darling dla {n} danych o odchyleniu {sd}:", power_anderson_darling)
            print(f"Moc testu Lilliefors dla {n} danych o odchyleniu {sd}", power_lilliefors)
            print(f"Moc testu Jarque_Bera dla {n} danych o odchyleniu {sd}", power_jarque_bera)


def gamma_distribution_power_test(list_number_of_datas,shape,scale):
    iterations = 1000
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







number_of_data = [10,50,500,1000,5000]
std = [1,3,5,10,20,50]
list_of_degrees_of_freedom = [2,5,10,50,100,500,1000,5000]
shape = [1,2,5,10,20]
scale = [1,2,4,8,12]


#normal_distribution_power_test(Normality_test, NormalDistribution, check_test_power)
#t_student_distribution_power_test(number_of_data, list_of_degrees_of_freedom)
#normal_distribution_power_test(number_of_data, std, False)
gamma_distribution_power_test(number_of_data,shape,scale)
