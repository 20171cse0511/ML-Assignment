import numpy as np 
import pandas as pd 
df=pd.read_csv('i.csv")


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X=df[['tid']]
Y=df[['year']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
p1 = dt.predict([[93779]])
p1[0]
