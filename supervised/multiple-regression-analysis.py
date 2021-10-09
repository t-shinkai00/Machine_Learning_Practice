import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
dataset=load_boston()
# print(dataset)

x, t, columns=dataset.data, dataset.target, dataset.feature_names
# print(type(x),x.shape)
# print(type(t),t.shape)

df=pd.DataFrame(x,columns=columns)
df["Target"]=t
# print(df)