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

# データフレーム作成
df=pd.DataFrame(x,columns=columns)
df["Target"]=t
# print(df)

# 入力と出力の切り分け
t=df["Target"].values
x=df.drop(labels=["Target"], axis=1).values

# 学習用データとテストデータ切り分け
from sklearn.model_selection import train_test_split

x_train, x_test, t_train, t_test=train_test_split(x, t, test_size=0.3, random_state=0)

# 学習モデルのインポート
from sklearn.linear_model import LinearRegression

model=LinearRegression()
# print(model)

# 学習
model.fit(x_train, t_train)