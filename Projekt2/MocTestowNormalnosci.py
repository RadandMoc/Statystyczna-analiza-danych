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
dane = DATA[DATA['Telefon\zmienne'].str.startswith("Apple")]
daneSAMSUNG = DATA[DATA['Telefon\zmienne'].str.startswith("SAMSUNG")]["Cena"]
daneHuawei = DATA[DATA['Marka'] == "Huawei"]["Cena"]
#Czarny = DATA[DATA['Kolor'] == "Lawendowy"]["Cena"]

print(check_test_power(Normality_test.Jarque_Bera,dane["Cena"]))
print(check_test_power(Normality_test.Jarque_Bera,daneSAMSUNG))
print(check_test_power(Normality_test.Jarque_Bera,daneHuawei))


#print(check_test_power(Normality_test.Jarque_Bera,Czarny))

#stat, p = bartlett(dane, daneSAMSUNG, daneHuawei)
#print(f"Nie ma podstaw do odrzucenia hipotezy zerowej p-value >0.05 {p>0.05} wartosc wynosi {p}")


#stat, p = levene(dane, daneSAMSUNG, daneHuawei)

#print("Statystyka testu:", stat)
#print("P-wartość:", p)





def change_group_color(color):
    black_colors = [
    "Black", "Moonlight Black", "Electric Black", "Ink Black", "Mystery Black", "Matte Black",
    "Stellar Black", "Crystal Black", "Midnight Black", "JET BLACK", "Luminous Black",
    "Prism Black", "Piano Black", "Lightening Black", "Marble Black", "Dazzling Black",
    "Twilight Black", "Diamond Black", "Jade Black", "Charcoal Black", "Aurora Black",
    "Universe Black", "Supersonic Black", "Infinite Black", "Cyber Black", "Mystic Black",
    "Prism Crush Black", "Caviar Black", "Cosmic Black", "Absolute black", "Ceramic Black",
    "Black Sapphire", "Phantom Black", "Aura Black", "Graphite Black", "Space Black",
    "Metallic Black", "Genuine Leather Black", "Titan Black", "Black & Blue", "Black Gold",
    "Gold & Black", "Carbon Black", "Slate Black", "Black Ninja", "Black Blue",
    "Ambitious Black", "Lightning Black", "Cosmos Black", "Noble Black", "Meteorite Black",
    "Stealth Black", "Black Diamond", "Berlin Gray", "Charcoal Grey", "Slate Grey", "Black Pearl"
    ]
    white_colors = [
    "White","Silky White", "Starry Night", "Dazzling White", "Mist White", "Prism White",
    "Ivory White", "Pearl White", "Arctic White", "Snow White", "Frost White",
    "Cloud White", "Pure White", "Angel White", "Ghost White", "Moonlight White",
    "Alpine White", "Ceramic White", "Coral White", "Fantasy White", "Glacier White", "Mint Cream"
    ]
    blue_colors = [
    "Deep Blue", "Blue", "Flowing Silver", "Crystal Blue", "Lake Green",
    "Fantastic Purple", "Magic Blue", "Starry Blue", "Navy Blue", "Fancy Blue",
    "Pearl Blue", "Astral Blue", "Mystery Blue", "Sky Blue", "Midnight Blue",
    "Glaring Gold", "Ocean Blue", "Cosmos Blue", "Neptune Blue", "Royal Blue",
    "Dark Blue", "Oxford Blue", "Rainbow Black", "Blue Coral", "Prism Blue"
    ]
    red_colors = [
    "Red", "Bordeaux Red", "Maroon Red", "Ferrari Red", "Ruby Red", "Garnet Red",
    "Flame Red", "Rose Red", "Sunset Red", "Crimson Red", "Wine Red",
    "Cherry Red", "Blazing Red", "Radiant Red", "Lava Red", "Fire Red"
    ]
    green_colors = [
    "Green", 
    "Emerald Green", "Jade Green", "Moss Green", "Olive Green", "Sea Green",
    "Forest Green", "Mint Green", "Lime Green", "Neon Green", "Jungle Green",
    "Sage Green", "Pine Green", "Apple Green", "Kelly Green", "Hunter Green"
    ]
    gold_colors = [
    "Rose Gold", "Sunrise Gold", "Champagne Gold", "Satin Gold", "Harvest Gold",
    "Honey Gold", "Amber Gold", "Rusty Gold", "Bronze Gold", "Desert Gold",
    "Golden", "Luxury Gold", "Elegant Gold", "Rich Gold", "Royal Gold", "Gold"
    ]
    silver_colors = [
    "Silver", "Metallic Silver", "Platinum Silver", "Silky Silver", "Chrome Silver", "Steel Silver",
    "Titanium Silver", "Iron Silver", "Moon Silver", "Galactic Silver", "Cosmic Silver",
    "Starlight Silver", "Glacier Silver", "Polar Silver", "Graphite Silver", "Mystic Silver", "Grey"
    ]

    
    if color in black_colors:
        return "Black"
    elif color in white_colors:
        return "White"
    elif color in blue_colors:
        return "Blue"
    elif color in red_colors:
        return "Red"
    elif color in green_colors:
        return "Green"
    elif color in gold_colors:
        return "Golden"
    elif color in silver_colors:
        return "Silver"
    else:
        return color





colors = dane['Kolor'].unique()
color_groups = [dane[dane['Kolor'] == color]['Cena'] for color in colors]
anova_result = f_oneway(*color_groups)
print(anova_result)

data_phone = pd.read_csv('Flipkart_Mobiles.csv',sep=",",decimal=".")
data_renamed = data_phone.rename(columns={'Selling Price': 'Selling_Price'})


data_renamed['General_Color'] = data_renamed['Color'].apply(change_group_color)

pd.set_option('display.max_rows', None)  # Ustawienie na None wyświetli wszystkie wiersze
pd.set_option('display.max_columns', None)  # Ustawienie na None wyświetli wszystkie kolumny
pd.set_option('display.width', None)  # Ustawienie szerokości wyświetlania, aby uniknąć zawijania wierszy
pd.set_option('display.max_colwidth', None)  # Ustawienie maksymalnej szerokości kolumn


print(data_renamed[~data_renamed["General_Color"].isin(["Black", "White", "Red", "Green", "Silver", "Golden", "Blue"])]["General_Color"].unique())

"""
model = ols('Selling_Price ~ Color', data=data_renamed).fit()
anova_result = sm.stats.anova_lm(model, typ=1)
print(anova_result)
"""