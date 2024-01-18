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
    'Black', 'Moonlight Black', 'Electric Black', 'Ink Black', 'Mystery Black', 'Matte Black',
    'Stellar Black', 'Crystal Black', 'Midnight Black', 'JET BLACK', 'Luminous Black',
    'Prism Black', 'Piano Black', 'Lightening Black', 'Marble Black', 'Dazzling Black',
    'Twilight Black', 'Diamond Black', 'Jade Black', 'Charcoal Black', 'Aurora Black',
    'Universe Black', 'Supersonic Black', 'Infinite Black', 'Cyber Black', 'Mystic Black',
    'Prism Crush Black', 'Caviar Black', 'Cosmic Black', 'Absolute black', 'Ceramic Black',
    'Black Sapphire', 'Phantom Black', 'Aura Black', 'Graphite Black', 'Space Black',
    'Metallic Black', 'Genuine Leather Black', 'Titan Black', 'Black & Blue', 'Black Gold',
    'Gold & Black', 'Carbon Black', 'Slate Black', 'Black Ninja', 'Black Blue',
    'Ambitious Black', 'Lightning Black', 'Cosmos Black', 'Noble Black', 'Meteorite Black',
    'Stealth Black', 'Black Diamond', 'Berlin Gray', 'Charcoal Grey', 'Slate Grey', 'Black Pearl', 'CRYSTAL BLACK', 
    'Dynamic Black', 'Fluid Black', 'Mystery Black', 'Jet Black',  ' Tornado Black','Just Black'
    'Midnight Black', 'Carbon Black', 'Piano Black', 'Phantom Black', 'Aurora Black', 'Graphite Black', 
    'Charcoal Black', 'Eclipse Black', 'Meteorite Black', 'Onyx Black', 'Cosmic Black', 'Stealth Black', 
    'Predator Black', 'Asteroid Black', 'Interstellar Black', 'Nebula Black', 'Galactic Black', 
    'Night Black', 'Space Black', 'Shadow Black', 'Starry Night Black', 'Black', 'Jet Black', 'Midnight Black',
    'Slate Black', 'Night Black', 'Cosmic Black', 'Kind of Grey', 'Metallic Grey', 
    'Pitch Black', 'Two shades of black', 'Matrix Purple', 'Neo Black', 'Stardust Black',
    'Shadow Grey', 'Starlight Black', 'Blazing Black','Starry Black','Mist Black','Glowing Black','Mirror Black',
    'Thunder Black','Tornado Black','Quartz Black','Aether Black','Milan Black','Polar Black','Rainbow Black'
    ,'Sonic Black','Black Titan', 'Laser Black', 'Sword Black', 'Blade Silver',
    'Fusion Black', 'Fusion Green', 'Dashing Black', 'Racing Black', 'Laser Green',
    'Galaxy Black', 'Galaxy Green', 'Sun Black', 'Night Black', 'Nova Black', 'Cosmic Black',
    'Star Black', 'Chrome Black', 'Power Black', 'Midnight Black', 'Carbon Black', 'Steel Black',
    'Obsidian Black', 'Olive Black', 'Electric Black', 'Titanium Black', 'Gravity Black', 'Black Leather',
    'Power Black', 'Mighty Black', 'Fantastic Black', 'Emerald Black', 'Vinyl Black', 'Cosmic Black',
    'Moonlight Black', 'Nebula Black', 'Astro Moonlight White', 'Electric Black', 'Prime Black','Indigo Black',
     'Brilliant Black', 'Green and Greener', 'Out of Blue', 'Slate Blue', 'Armoured Edition', 'Enigma Black','Cross Black',
     'Black&Blue', 'Copper/Black', 'Black & Copper', 'Mirage Black', 'Awesome Black', 'Super Black', 'Black/Tuxedo Black', 'Venom Black',
     "Just Black", "Carbon", "Charcoal", "Dark Grey", "Denim Black", "Celestial Black", "Deep Black", "Raven Black", "Charcoal Gray", 
    "GREY/BLACK", "Dark Night", "Cosmic Black", "Roman Black", "Luna White", "Pearl Black", "Gunmetal Grey", "Glossy Black",
    "Cosmic Black", "Charcoal", "Black Onyx", "Black Sapphire", "Grey/Black",
    "CHARCOAL", "Prism Dot Black", "black sapphire", "Black Sky", "COSMIC BLACK",
     'Lavender  ', 'Alpha Grey'
    ]
    white_colors = [
    'White','Silky White', 'Starry Night', 'Dazzling White', 'Mist White', 'Prism White',
    'Ivory White', 'Pearl White', 'Arctic White', 'Snow White', 'Frost White',
    'Cloud White', 'Pure White', 'Angel White', 'Ghost White', 'Moonlight White',
    'Alpine White', 'Ceramic White', 'Coral White', 'Fantasy White', 'Glacier White', 
    'Mint Cream', 'Pearl White', 'Fairy White', 'Sky White', 'Clearly White',
    'Unicorn White', 'Ivory White', 'Snow White', 'Cloud White', 'Misty White', 'Lunar White', 
    'Moonlight White', 'Frost White', 'Polar White', 'Super Polar White', 'Arctic White', 
    'Ghost White', 'Starlight White', 'Cosmic White', 'Glacier White', 'Celestial Snow', 'Classic White',
    'White', 'Metallic White', 'Fancy White', 'Stream White', 'Vanilla Mint', 'Fancy white', 'Jewelry White', 'Stardust White',
    'Aurora Silver', 'Aurora Gray', 'Illusion Sky', 'Crystal White', 'Genuine Leather Brown', 'White Knight','So White', 'That White',
    "Not Pink", "Chroma White", "Comet White", "Starlight", "Midnight", "Diamond White", "Prism Crush White", "White Pearl", "Aura Glow", 
    "White Frost", "Pearl white", "Chic White", "Marble White",  'MOONLIGHT WHITE','Copper White',
    "More Than White", "White Birch", "Santorini White", "White", "Garlic",
    "Luna White","Awesome White", "White & Copper", "Cloud White", "Cream",
    "Dawn White", "Moonlight White", "Luna White", "Sprinkle White", 'Luna White ',  'White  '
    ]
    blue_colors = [
    'Deep Blue', 'Blue', 'Flowing Silver', 'Crystal Blue', 'Lake Green', 'Quetzal Cyan'
    'Fantastic Purple', 'Magic Blue', 'Starry Blue', 'Navy Blue', 'Fancy Blue','Energetic Blue',
    'Pearl Blue', 'Astral Blue', 'Mystery Blue', 'Sky Blue', 'Midnight Blue', 'Celestial Magic',
    'Glaring Gold', 'Ocean Blue', 'Cosmos Blue', 'Neptune Blue', 'Royal Blue',
    'Dark Blue', 'Oxford Blue', 'Blue Coral', 'Prism Blue', 'Deep Blue', 'Electric Blue',
    'Navy Blue', 'Ocean Blue', 'Sapphire Blue', 'Celestial Blue', 'Space Blue', 'Arctic Blue', 'Sky Blue', 
    'Midnight Blue', 'Neptune Blue', 'Royal Blue', 'Aurora Blue', 'Aegean Blue', 'Ocean Wave','Aquamarine Blue','Glowing Galaxy', 
    'Bolivia Blue', 'Auroral Blue', 'Diamond Blue', 'Twilight Blue', 'Galactic Blue',  
    'Ocean Blue', 'Blue MB', 'Tornado Blue', 'Amazing Silver', 'Thunder Blue', 'Blue MB', 'Blue & Silver', 'Mirage Blue', 'Denim Blue', 'Mirage Blue', 'Sky Blue', 'Milkyway Grey', 'Iceberg Blue', 'IceBerg Blue', 'Mirage Blue', 'Mirage Blue', 'Azure Glow', 'Blazing Blue', 'Light Blue', 
    'Laser Grey', 'Awesome Blue', 'Awesome Blue', 'Mirage Blue', 'Mirage Blue', 'Azure Glow', 'Blazing Blue', 'Laser Grey', 'Awesome Blue','Ice',
    'MAGIC BLUE', 'Thunder blue','Glaze Blue','Glacier Blue','midnight blue','Moroccan Blue', 'Icy Blue','Sunrise Blue ','Startrails Blue','Charcoal Blue',
    'Sonic Blue','Skyline Blue','Frost Blue','Laser Blue', 'Aqua Blue', 'Cool Blue', 
    'Racing Blue', 'Watery Blue', 'Supersonic Blue', 'Power Blue', 'Infinite Blue', 
    'Mist Blue', 'Frozen Blue', 'Oxygen Blue', 'Sparkling Blue', 'Dashing Blue', 'Cosmic Blue', 'Seawater Blue', 
    'Cloud Blue', 'Sapphire Gradient', 'Nebula Blue', 'Twilight Blue', 'Dawn Blue', 'Neon Blue', 'Blue Lagoon', 'Ice Blue',"Metallic Blue"
    'Cool Blue', 'Racing Blue', 'Seawater Blue', 
    'Iceberg blue', 'Lemonade Blue', 'Blue Lagoon', 'Nebula Blue', "Gradation Blue","Coral Blue",
    'Twilight Blue', 'Dawn Blue', 'Neon Blue', 'Awesome Blue', 'Ocean Blue', "Pastel Sky", "Misty Blue",
    'Azure Blue', 'Electric Blue', 'Oceanic Blue', 'Midnight Blue', 'Sapphire Blue',
    'Deep Blue', 'Cerulean Blue', 'Sky Blue', 'Steel Blue', 'Caribbean Blue', 'Dark Blue', 'Turquoise Blue', 
    'Royal Blue', 'Indigo Blue', 'Cyan Blue', 'Lavender Blue', 'Pacific Blue', 'Baby Blue', 'Mystic Blue', 'Prism Blue', 'Crystal Blue',
    'Starlight Blue', 'Ultramarine Blue', 'Elegant Blue', 'Dreamy Blue', 'Wave Blue', 'Ocean Wave Blue', 'Frost Blue', 
    'Breeze Blue', 'Aqua Blue', 'Pastel Sky Blue', 'Glacier Blue', 'Ice Blue', 'Epic Blue', 'Dream Blue', 'Cool Ocean Blue', 
    'Sapphire Gradient Blue', 'Indigo Gradient Blue', 
    'Dusk Blue', 'Daybreak Blue', 'Moonlight Blue', 'Cosmic Blue', 'Cosmic Gray Blue',
    'Deep Sea Blue', 'SeaBlue', "Cross Blue", "Universe Blue", "Comet Blue", "Fusion Blue", "Victory Blue", 
    "That Blue", "So Blue", "Lightning Blue", "Diamond Sapphire", 
    "Radiant Blue", "Fjord Blue", "Nordic Blue", "Night | Dark Blue", 
    "Arctic blue", "Sky blue", "Prism Crush Blue", "Cloud Navy", 
    "Awesome Blue", "Metallic Blue", "Prism Crush Violet", "Sierra Blue", 
    "Polished Blue", "Cosmic Gray", "Prism Dot Gray", "Dark Gray", 
    "Iris Charcoal", "Atlantic Blue", "Pacific Sunrise", "Jazz Blue", 
    "Not just Blue", "Lake Blue", "Sea Blue", "Rainbow Blue", "Neo Blue", 
    "Voyager Grey", "Aurora Dawn","Tempered Blue", "Tahiti Blue", "Midnight Blue", "Fjord", "Mostly Blue", 
    "Purist Blue", "Shimmer Blue", "Starry Glow", "Midnight Jazz", "Cosmic Blue"
    ]
    red_colors = [
    'Red', 'Bordeaux Red', 'Maroon Red', 'Ferrari Red', 'Ruby Red', 'Garnet Red',
    'Flame Red', 'Rose Red', 'Sunset Red', 'Crimson Red', 'Wine Red',
    'Cherry Red', 'Blazing Red', 'Radiant Red', 'Lava Red', 'Fire Red', 'Sunrise Red', 'Radiant Red',
    'Fire Red', 'Lava Red', 'Ruby Red', 'Cherry Red', 'Coral Red', "Cloud Red", "Rich Cranberry"
    'Rose Red', 'Blazing Red', 'Sunset Red', 'Crimson Red', 'Wine Red', 'Mars Red', 'Flame Red', 'Bordeaux Red','Bordeaux Red ',
    'Red Brick', 'Rust Red', 'Lightning Red', 'Fiery Gold', 'Lightning Orange', 'Copper', 'Pink', 'Rose Pink', 'Brave Blue',
    'Lavender Violet', 'Champagne', 'Blush Gold', 'Magic Gold', 'Amber Red', 'Fervor Red',
    'Diamond Red','Red Sunset', 'Red Wine', 'Cherry Red', 'Ruby Red', 'Flame Red', 
    'Crimson Red', 'Scarlet Red', 'Burgundy Red', 'Candy Red', 'Rouge Red', 'Fiery Red', 'Bold Red', 'Rosso Red', 'Ruby Red',
    'Bright Red', 'Burgundy Red', "Solar Red", "Diamond Ruby", "Warm Red", "Prism Crush Red", "Aura Red", 
    "Phantom Red", "Pheonix Red", "Neon Spark", "Agate Red", 
    "Nebula Red", "Brick Red","Nebula Red", "Phantom Red", "Sunrise Flare", "Fantastic Rainbow"
    ]
    green_colors = [
    'Green', 'Aurora'
    'Emerald Green', 'Jade Green', 'Moss Green', 'Olive Green', 'Sea Green',
    'Forest Green', 'Mint Green', 'Lime Green', 'Neon Green', 'Jungle Green',
    'Sage Green', 'Pine Green', 'Apple Green', 'Kelly Green', 'Hunter Green', 
    'Mint Green', 'Emerald Green', 'Jade Green', 'Aurora Green', 'Haze Green',
    'Neon Green', 'Lime Green', 'Olive Green', 'Jungle Green', 'Sea Green', 
    'Tropical Green', 'Forest Green', 'Hunter Green', 'Moss Green','Phantom Green',
    'Meadow Green', 'Pine Green','Marine Green', 'Galactic Blue', 'Marble Green', 
    'Jungle Green', 'Ocean Green', 'Oxygen Green', 'Pearl Green', 'Awesome Mint',
    'Nature Green', 'Camo Green','Green Wave',
    'Morandi Green', 'City Blue', 'Alpine Green', 'Azure Glow', 'Glowing Green','Aurora','Moonlight Jade'
    "Rich Green", "Crystal Green", "That Green", "Cyan Green", "Mystic Green", 
    "Quartz Green", "Pebble Blue", "Cloud Mint", "Electric Green", 
    "Midnight Green", "Bright Green", "Mint", "CYAN",
    "Aqua Green", "Neon Green","Rich Green", "Aqua Green", "Neon Green", "Aquamarine Green", "Coral Green", "Electric Green"
    
    ]
    gold_colors = [
    'Rose Gold', 'Sunrise Gold', 'Champagne Gold', 'Satin Gold', 'Harvest Gold',
    'Honey Gold', 'Rusty Gold', 'Bronze Gold', 'Desert Gold',
    "Sun Kissed Leather", "Titan", "Titanium Sapphire", "Mystic Bronze",
    'Golden', 'Luxury Gold', 'Elegant Gold', 'Rich Gold', 'Royal Gold', 'Gold', 'Moonlight Gold', 'Sunrise Gold', 
    'Champagne Gold', 'Harvest Gold', 'Bronze Gold', 'Amber Gold', 'Honey Gold', 
    'Rich Gold', 'Glowing Gold', 'Royal Gold', 'Dazzling Gold',
    'Saffron Gold', 'Luxury Gold', 'Elegant Gold', 'Sunset Gold', 'Desert Gold', 'Metallic Gold', 'Gold',
    'Majestic Gold', 'Polar Gold', 'CHAMPAGNE GOLD', 'Latte Gold', 'Gold Sand', 'Angel Gold', 
    'Gold Platinum', 'Mocha Gold', 'Black and Gold', 'Gold Sepia', 'Gold and Silver', 'Sunshine Gold',
    'Sunset Jazz', 'Sunset Melody', 'Sunset Dazzle', 'Sunset Flare', 'Sunset Blue', 'Golden','Bronze Gold Black','Serene Gold',
    "Titanium", "Frosted Gold", "Phantom White", "Copper Gold", "Maple Gold", 
    "Topaz Gold", "Matte Gold", "Fine Gold", "dark gold","Polished Copper", "Copper Gold", "Metallic Copper", "Gold"
    
    ]
    silver_colors = [
    'Silver', 'Metallic Silver', 'Platinum Silver', 'Silky Silver', 'Chrome Silver', 'Steel Silver',
    'Titanium Silver', 'Iron Silver', 'Moon Silver', 'Galactic Silver', 'Cosmic Silver',
    'Starlight Silver', 'Glacier Silver', 'Polar Silver', 'Graphite Silver', 'Mystic Silver', 'Grey',
    'Metallic Silver', 'Moonlight Silver', 'Stainless Silver', 'Chrome Silver', 'Silver Wave'
    'Steel Silver', 'Twilight Grey', "Lunar Gray",
      'Titanium Silver', 'Graphite Silver', 'Quantum Silver', 'Smokey Gray',
    'Iron Silver', 'Silver Grey', 'Cosmic Silver', 'Galactic Silver', 'Very Silver',
    'Starlight Silver', 'Polar Silver', 'Glacier Silver', 'Silver Diamond', 'Tradew Grey',
    'Metallic White', 'Crystal Silver', 'Space Silver', 'Classic Silver', 'Platinum', 'Grey '
    'Titan', 'Silver Titan', 'Platinum Grey', 'Metallic Gray', 'Titan Silver', 'Silver Blue',
    'Stainless Black', 'Shimmery White', 'Sterling Blue', 'Dark Pearl', 'Frosted Silver',
    'Frosted Pearl', 'Polished Silver', 'Prism Magic', 'Crystal Symphony', 'Phantom Gray',
    'Lunar Grey', 'LUNAR WHITE', 'CELESTIAL SILVER', 'Polaris Blue', 'Silver Titanium', 
    "Cool Grey", "Metal Grey", "Grey / Silver", "Space Grey", "Space Gray", "Iron", "Steel",
    "Pebble Grey", "Frosted Champagne", "Electric Graphite", "Polished Graphite", "Slate Gray", "Midnight Grey",
    'Gold Platinum', 'Aura White', 'Phantom Silver', 'Matte Aqua', 'Topaz Blue','Rainbow Silver', 'Orchid Grey',
    'Quetzal Cyan', 'Symphony Cyan', 'Glacier Green', 'Mithril Grey', 'Nordic Secret', 'Sandstone Black',
    'Sapphire Cyan', 'Shadow Grey', 'Starlight Black', 'Storm White', 'Meteor Black', 'Stargaze White',
    'Rainbow Silver', 'Waterfall Grey','Shark Grey','AURORA SILVER', 'Meteor Silver', 'Glory Silver', 'Silver White', 
    'Prism Crush Silver', 'Phantom Silver', 'Sonic Silver', 'Stone Silver', 'Silver Feather', 'Piano Black/Silver', 
    'Flash Silver', 'Turquoise', 'Silver Armor', 'Silver Charm', 'Silver Spoon', 'Silver Bullet', 'Space Silver',
    'Silverstar', 'Cosmic Silver', 'Silver Dust', 'Ceramic Silver', 'Silver Strand', 'Polished Silver', 'Silver Shine', 
    'Silver Lining', 'Silver Mist', 'Silver Diamond', 'Silver Moon', 'Silver Galaxy', 'Silver Linen', 'Silver Dawn',
    'Silver Ray', 'Silver Frost', 'Silver Smoke', 'Silver Dusk', 'Silver Mirage', 'Silver Spark', 'Silver Arrow',
    'Silver Blade', 'Silver Comet', 'Silver Dream', 'Silver Flash', 'Silver Fox', 'Silver Fusion', 'Silver Ghost', 
    'Silver Glow', 'Silver Jewel', 'Silver Magic', 'Silver Night', 'Silver Shadow', 'Silver Sky', 'Silver Sparkle',
    'Silver Star', 'Silver Storm', 'Silver Sun', 'Silver Wave', "Racing Silver", "Watery Grey", "Cyber Silver", "Power Silver", 
    "Haze Crush Silver", "Mercury Silver","Pewter / White", "Metallic Silver", "Lunar Silver", "Celestial Silver", "Gunmetal Silver",
    'Meteor Grey', 'Saffron Grey', "Grey ", "Rich Grey", 'Slate Gray' 'Twilight Grey', "Titan Gray", 'Aurora Grey', "Dynamic Gray",  "Volcanic Grey"
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

data_phone = pd.read_csv('PhoneData.csv',sep=";")
data_renamed = data_phone.rename(columns={'Selling Price': 'Selling_Price'})


data_renamed['General_Color'] = data_renamed['Color'].apply(change_group_color)

pd.set_option('display.max_rows', None)  # Ustawienie na None wyświetli wszystkie wiersze
pd.set_option('display.max_columns', None)  # Ustawienie na None wyświetli wszystkie kolumny
pd.set_option('display.width', None)  # Ustawienie szerokości wyświetlania, aby uniknąć zawijania wierszy
pd.set_option('display.max_colwidth', None)  # Ustawienie maksymalnej szerokości kolumn


array = data_renamed.to_numpy()


print(cut_data_to_the_same_size(array))



#print(data_renamed[~data_renamed["General_Color"].isin(["Black", "White", "Red", "Green", "Silver", "Golden", "Blue"])]["General_Color"].unique())
#print(len(data_renamed[~data_renamed["General_Color"].isin(["Black", "White", "Red", "Green", "Silver", "Golden", "Blue"])]["General_Color"].unique()))

"""
model = ols('Selling_Price ~ Color', data=data_renamed).fit()
anova_result = sm.stats.anova_lm(model, typ=1)
print(anova_result)
"""