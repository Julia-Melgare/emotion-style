import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
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
print(output)

#import data - Posed datasets
names = ['Andre', 'Conrado', 'Diogo', 'Gabriel', 'Julia', 'Maurer', 'Paulo', 'Pedro', 'Rovane', 'Victor', 'Wetzel']

lab = []
for name in names:
    df = (pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\Emotion Style CSVs" + "\\" + name + r" All.csv",header=None, skiprows=2))
    lab.append(df.iloc[[0]]) #emotion
df1 = pd.concat(lab)

facesDB = []
df2 = pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\FacesDB-All.csv", header=None)
emotion = 0  #emotion
for i in range(0,36):
    facesDB.append(df2.iloc[[emotion]])
    emotion += 6
df2 = pd.concat(facesDB)

df = pd.concat([df1, df2])
df = df.iloc[:, :-1]
df.reset_index(drop=True, inplace=True)
#print(df)

# Import data - Spontaneous dataset
# path = r'Users\jujum\OneDrive\Documents\IC\datasetSpont\Julia_CSVs1'
# directory = os.path.join("c:\\",path)

# folder1 = []
# for root,dirs,files in os.walk(directory):
#     for file in files:
#        if file.endswith(".csv"):
#            sub = pd.read_csv(directory+"\\"+file, header=None, skiprows=1)
#            del sub[0]
#            folder1.append(sub)
# df1 = pd.concat(folder1)

# path = r'Users\jujum\OneDrive\Documents\IC\datasetSpont\Julia_CSVs2'
# directory = os.path.join("c:\\",path)

# folder2 = []
# for root,dirs,files in os.walk(directory):
#     for file in files:
#        if file.endswith(".csv"):
#            sub = pd.read_csv(directory+"\\"+file, header=None, skiprows=1)
#            del sub[0]
#            folder2.append(sub)
# df2 = pd.concat(folder2)

# df = pd.concat([df1, df2])

X = df #pd.DataFrame(preprocessing.scale(df),columns = df.columns)

# Recreate clusters
num_clusters = max(numlist)+1
scount = 0
clusters = [[] for i in range(num_clusters)]
for num in numlist:
    clusters[num].append(tuple(X.iloc[scount].values.tolist()))
    scount+=1
print(clusters[0])

#cnum = 0
# diff_clusters = [[] for i in range(num_clusters)]
# for cluster in clusters:
#     #print("C"+str(cnum))
#     centroid = tuple([sum(y) / len(y) for y in zip(*cluster)])
#     #print(centroid)
#     closest = float("inf")
#     for member in cluster:
#         diff = scipy.spatial.distance.euclidean(member,centroid)
#         diff_clusters[cnum].append(diff)
#         #print(str(cluster.index(member))+": "+str(diff))
#         if(closest > diff):
#             closest = diff
#     cnum+=1
#     #print("closest = "+str(closest))

#for i in range(0, num_clusters):
#    x = dict(zip(sub_clusters[i], diff_clusters[i]))
#    print({k: v for k, v in sorted(x.items(), key=lambda item: item[1])})




        


