import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

sns.set()

#import data
names = ['Andre', 'Conrado', 'Diogo', 'Gabriel', 'Julia', 'Maurer', 'Paulo', 'Pedro', 'Rovane', 'Victor', 'Wetzel']

lab = []
for name in names:
    df = (pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\Emotion Style CSVs" + "\\" + name + r" All.csv",header=None, skiprows=2))
    lab.append(df.iloc[[4]]) #emotion: 0 - happiness   1 - fear   2 - disgust     3 - anger   4 - surprise    5 - sadness
df1 = pd.concat(lab)

facesDB = []
df2 = pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\FacesDB-All.csv", header=None)
emotion = 2  #emotion: 0 - happiness    1 - sadness     2 - surprise    3 - anger    4 - disgust     5 - fear
for i in range(0,36):
    facesDB.append(df2.iloc[[emotion]])
    emotion += 6
df2 = pd.concat(facesDB)

path = r'Users\jujum\OneDrive\Documents\IC\datasetSpont\Julia_CSVs1'
directory = os.path.join("c:\\",path)

folder1 = []
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           sub = pd.read_csv(directory+"\\"+file, header=None, skiprows=1)
           del sub[0]
           folder1.append(sub)
df3 = pd.concat(folder1)
df3.columns = range(df3.shape[1])

path = r'Users\jujum\OneDrive\Documents\IC\datasetSpont\Julia_CSVs2'
directory = os.path.join("c:\\",path)

folder2 = []
for root,dirs,files in os.walk(directory):
    for file in files:
       if file.endswith(".csv"):
           sub = pd.read_csv(directory+"\\"+file, header=None, skiprows=1)
           del sub[0]
           folder2.append(sub)
df4 = pd.concat(folder2)
df4.columns = range(df4.shape[1])

df = pd.concat([df1, df2, df3, df4])
df = df.iloc[:, :-1]
print(df)
#df.reset_index(drop=True, inplace=True)
X = pd.DataFrame(preprocessing.scale(df),columns = df.columns)

#GMM
#n_components = np.arange(2, 9)
#models = [GaussianMixture(n, covariance_type='full', random_state=4).fit(X) for n in n_components] #4
#plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
#plt.plot(n_components, [silhouette_score(X, m.predict(X)) for m in models])
#plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
#plt.legend(loc='best')
#plt.xlabel('Number of Clusters')
#plt.ylabel('Silhouette Score')

#plt.show() #2 to 8 clusters

# gmm3 = GaussianMixture(n_components=3, random_state=6)
# gmm3.fit(X)
# print(gmm3.predict(X))

# save models using pickle
# filename ='C:\\Users\\jujum\\OneDrive\\Documents\\IC\\datasetSpont\\models\\gmm-3-global.sav'
# pickle.dump(gmm3, open(filename, 'wb'))

#KMEANS

n_components = np.arange(2, 9)
models = [KMeans(n_clusters=n).fit(X) for n in n_components] #4 w/ seed 1 - 3 w/ seed 5
plt.plot(n_components, [silhouette_score(X, m.predict(X)) for m in models])
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

plt.show()

# kmeans2 = KMeans(n_clusters=2, random_state=5)
# kmeans2.fit(X)
# print(kmeans2.predict(X))


# save models using pickle
# filename ='C:\\Users\\jujum\\OneDrive\\Documents\\IC\\datasetSpont\\models\\kmeans-2-global.sav'
# pickle.dump(kmeans2, open(filename, 'wb'))
