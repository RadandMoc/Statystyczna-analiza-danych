import pandas as pd
import numpy as np

print("Hello World")

readData=pd.read_csv("C:/Users/zapar/Python/BOT/Statystyczna-analiza-danych/DaneTelefonow.csv",sep=";")
numpy_array2 = readData.iloc[:,1:].to_numpy()
print(numpy_array2)