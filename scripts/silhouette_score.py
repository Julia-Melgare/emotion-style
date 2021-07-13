import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy

input = sys.argv[1]
numlist = list(map(int,input.split(' ')))
#print(numlist)

num_clusters = max(numlist)+1
scount = 1
sub_clusters = [[] for i in range(num_clusters)]
for num in numlist:
    sub_clusters[num].append('S'+str(scount))
    scount+=1

output = ''
for cluster in sub_clusters:
    #output+='{'
    for elem in cluster:
        if(cluster.index(elem) == len(cluster)-1):
            output+=str(elem)
        else:
            output+=str(elem)+', '
    output+='\n'#output+='}, '
#print(output)

#import data
names = ['Andre', 'Conrado', 'Diogo', 'Gabriel', 'Julia', 'Maurer', 'Paulo', 'Pedro', 'Rovane', 'Victor', 'Wetzel']

lab = []
for name in names:
    df = (pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\Emotion Style CSVs" + "\\" + name + r" All.csv",header=None, skiprows=2))
    lab.append(df.iloc[[1]]) #emotion: 0 - happiness   1 - fear   2 - disgust     3 - anger   4 - surprise    5 - sadness
df1 = pd.concat(lab)

facesDB = []
df2 = pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\FacesDB-All.csv", header=None)
emotion = 5  #emotion: 0 - happiness    1 - sadness     2 - surprise    3 - anger    4 - disgust     5 - fear
for i in range(0,36):
    facesDB.append(df2.iloc[[emotion]])
    emotion += 6
df2 = pd.concat(facesDB)

df = pd.concat([df1, df2])
df = df.iloc[:, :-1]
df.reset_index(drop=True, inplace=True)
#print(df)

X = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
silhouette_avg = silhouette_score(X, numlist)
print("The average silhouette_score is : ", silhouette_avg)
sample_silhouette_values = silhouette_samples(X, numlist)
print(sample_silhouette_values)
