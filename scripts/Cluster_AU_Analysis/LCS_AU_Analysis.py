import sys
import numpy as np
import pandas as pd
import scipy

def lcs(X, Y): 
    m=len(X)
    n=len(Y)
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
  
    # Fill d[][] in bottom up manner 
    for i in range(m + 1): 
        for j in range(n + 1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif X[i-1]  == Y[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
    return dp[m][n]

input = sys.argv[1]
numlist = list(map(int,input.split(' ')))

emotion = int(sys.argv[2])
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
    lab.append(df.iloc[[emotion]]) #emotion: 0 - happiness   1 - fear   2 - disgust     3 - anger   4 - surprise    5 - sadness
df1 = pd.concat(lab)

facesDB = []
df2 = pd.read_csv(r"C:\Users\jujum\OneDrive\Documents\IC\FacesDB-All.csv", header=None)
if emotion != 0:
    emotion = 6-emotion  #emotion: 0 - happiness    1 - sadness     2 - surprise    3 - anger    4 - disgust     5 - fear
for i in range(0,36):
    facesDB.append(df2.iloc[[emotion]])
    emotion += 6
df2 = pd.concat(facesDB)

df = pd.concat([df1, df2])
df = df.iloc[:, :-1]
df.reset_index(drop=True, inplace=True)

df.insert(0, "Subject", ["S"+str(i) for i in range(1,48)])
df.columns = ["Subject", "AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26"]
df['Cluster'] = numlist

clusters = []
for i in range(0, num_clusters):
    cluster = []
    cluster_df = (df.query('Cluster == '+str(i))) #filter by clusters
    for i in range(0, len(cluster_df.index)):
        subj_df = cluster_df.iloc[[i]]
        subj = []
        for au in df.columns[1:len(df.columns)-1]:
            if subj_df[au].iloc[0] > 0:
                subj.append(au)
        cluster.append(subj)
    clusters.append(cluster)

for i in range(0, num_clusters):
    cluster_i = clusters[i]    
    for j in range(0, num_clusters):
        cluster_j = clusters[j]
        sub_sum = 0
        counter = 0
        for sub_i in cluster_i:
            for sub_j in cluster_j:
                sub_sum += lcs(sub_i, sub_j)
                counter += 1
        avg_lcs = sub_sum/counter
        print("C"+str(i)+" & C"+str(j)+": "+str(avg_lcs))
    print("\n")






