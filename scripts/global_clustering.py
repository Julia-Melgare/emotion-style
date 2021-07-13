import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from sklearn import preprocessing

#import data
names = ['Andre', 'Conrado', 'Diogo', 'Gabriel', 'Julia', 'Maurer', 'Paulo', 'Pedro', 'Rovane', 'Victor', 'Wetzel']

subs = [[] for i in range(0,11)]
sub = 0

for name in names:
    df = (pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\Emotion Style CSVs" + "\\" + name + r" All.csv",header=None, skiprows=2))
    #df = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
    if name != 'Wetzel':
        df = df.iloc[:, :-1]
    for emotion in range(0,6):
        #print(tuple(df.iloc[emotion].values.tolist()))
        subs[sub].append(tuple(df.iloc[emotion].values.tolist()))
    sub+=1
#print(subs)

X = np.asarray([tuple(sub) for sub in subs])
#print(X)
nsamples, nx, ny = X.shape
d2_X = X.reshape((nsamples,nx*ny))

kmeans = KMeans(n_clusters=4)
kmeans.fit(d2_X)
#print(kmeans.predict(d2_X))

gmm = GaussianMixture(n_components=4)
gmm.fit(d2_X)
print(gmm.predict(d2_X))