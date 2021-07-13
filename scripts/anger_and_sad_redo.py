import sys
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy

#import data
names = ['Andre', 'Conrado', 'Diogo', 'Gabriel', 'Julia', 'Maurer', 'Paulo', 'Pedro', 'Rovane', 'Victor', 'Wetzel']

lab = []
for name in names:
    df = (pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\Emotion Style CSVs" + "\\" + name + r" All.csv",header=None, skiprows=2))
    if name != 'Wetzel':
        df = df.iloc[:, :-1]
    lab.append(df.iloc[[2]]) #emotion
df1 = pd.concat(lab)

facesDB = []
df2 = pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\FacesDB-All.csv", header=None)
emotion = 4  #emotion
for i in range(0,36):
    facesDB.append(df2.iloc[[emotion]])
    emotion += 6
df2 = pd.concat(facesDB)

df = pd.concat([df1, df2])
df = df.iloc[:, :-1]
df.reset_index(drop=True, inplace=True)
#print(df)

X = pd.DataFrame(preprocessing.scale(df),columns = df.columns).to_numpy()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print(kmeans.predict(X))