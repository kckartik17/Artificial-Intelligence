import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_excel('Data.xlsx')
X = dataset.iloc[:, :-1] 
y = dataset.iloc[:, -1] 