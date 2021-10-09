from pandas.core.algorithms import mode
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

t=df["Target"].values
x=df.drop(labels=["Target"], axis=1).values

from sklearn.model_selection import train_test_split

x_train, x_test, t_train, t_test=train_test_split(x, t, test_size=0.3, random_sate=0)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
# print(model)