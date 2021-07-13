import sys
import numpy as np
import pandas as pd
import scipy
from scipy import spatial
import matplotlib
import matplotlib.pyplot as plt
import random

def print_au_commands(au_list, values):
    i = 0
    for au in au_list:
        intensity = values[i]/100
        au_ctrl = ['','']
        if au in ['AU17', 'AU22', 'AU23', 'AU25']:
            au_ctrl[0] = '{}_U_CTRL.translateX'.format(au)
            au_ctrl[1] = '{}_D_CTRL.translateX'.format(au)
        elif au == 'AU26':
            au_ctrl[0] = '{}_CTRL.translateX'.format(au)
        else:
            au_ctrl[0] = '{}_L_CTRL.translateX'.format(au)
            au_ctrl[1] = '{}_R_CTRL.translateX'.format(au)
        cmd = 'setAttr \"{}\" {};\n'.format(au_ctrl[0], intensity)
        if au != 'AU26':
            cmd += 'setAttr \"{}\" {};\n'.format(au_ctrl[1], intensity)
        print(cmd)
        i+=1


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
    df = (pd.read_csv(r"D:\Users\jujum\OneDrive\Documents\IC\Emotion Style CSVs" + "\\" + name + r" All.csv",header=None, skiprows=2))
    lab.append(df.iloc[[emotion]]) #emotion: 0 - happiness   1 - fear   2 - disgust     3 - anger   4 - surprise    5 - sadness
df1 = pd.concat(lab)

facesDB = []
df2 = pd.read_csv(r"D:\Users\jujum\OneDrive\Documents\IC\FacesDB-All.csv", header=None)
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

au_clusters = []
df_clusters = []
for i in range(0, num_clusters):
    cluster = (df.query('Cluster == '+str(i))) #filter by clusters
    print("Cluster "+str(i))
    centroid = tuple(list(cluster.mean()[0:16]))
    centroid_distances = []
    for index, member in cluster.iterrows():
        centroid_distances.append(scipy.spatial.distance.euclidean(tuple(list(member[1:17])),centroid))
    cluster['Centroid_Distance'] = centroid_distances
    top_members = cluster.sort_values(['Centroid_Distance']).head(3) #3 most representative members only
    print('----- TOP MEMBERS -----')
    print(top_members)
    AU_freq = {}
    for au in df.columns[1:len(df.columns)-1]:
        au_presence = top_members.apply(lambda x: True if x[au] > 0 else False, axis = 1)
        AU_freq[au] = (len(au_presence[au_presence==True].index)/len(top_members.index))*100
    print(sorted(AU_freq.items(), key=lambda x: x[1], reverse=True)) # print(sorted(AU_freq, key=AU_freq.get, reverse=True))
    au_cluster = [x for x in AU_freq.keys() if AU_freq[x] == 100]
    au_clusters.append(au_cluster)
    df_clusters.append(top_members)
    #pd.set_option('display.max_columns', None)
    print(top_members.describe())


#plot data
SMALL_SIZE = 8
MEDIUM_SIZE = 18
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

au_colors_dict = {'AU1':'tab:blue', 'AU2':'tab:orange', 'AU4':'tab:green', 'AU5':'tab:red', 'AU6':'tab:purple', 'AU7':'tab:brown', 'AU9':'tab:pink', 'AU10':'tab:gray', 'AU12':'tab:olive', 'AU14':'tab:cyan', 'AU15':'salmon', 'AU17':'gold', 'AU20':'darkgreen', 'AU23':'teal', 'AU25':'mediumvioletred', 'AU26':'lightpink'}
labels = []
for i in range(0, num_clusters):
    labels.append('C'+str(i))

longest_au_len = max(len(elem) for elem in au_clusters)
means_list = []
stds_list = []
i = 0
for au_list in au_clusters:
    cluster_means = []
    cluster_stds = []
    for au in au_list:
        cluster_means.append(df_clusters[i].describe(include='all').loc['mean'].loc[au])
        cluster_stds.append(df_clusters[i].describe(include='all').loc['std'].loc[au])
    diff = longest_au_len - len(au_list)
    if diff > 0:
        for j in range(0, diff):
            cluster_means.append(0)
            cluster_stds.append(0)
    means_list.append(cluster_means)
    stds_list.append(cluster_stds)
    print('------------- CLUSTER {} -------------'.format(i))
    #print(au_list)
    #print(cluster_means)
    #print(cluster_stds)
    print_au_commands(au_list, cluster_means)
    i+=1
    
mean_bars = []
std_bars = []
for i in range(0, longest_au_len):
    mean_bars.append([x[i] for x in means_list])
    std_bars.append([x[i] for x in stds_list])

x = np.arange(len(labels))
width = 0.8

fig, ax = plt.subplots()
rects = []
offset = -0.25

flat_au_list = []
for i in range(0, longest_au_len):
    flat_au_list.append([x[i] for x in au_clusters if i < len(x)])

au_labels = []
for sublist in flat_au_list:
    for item in sublist:
        au_labels.append(item)

print(au_labels)
rect_i = 0
for rect in mean_bars:
    colors = []
    for v in rect:
        if v > 0:
            au = au_labels[rect_i]
            rect_i+=1
            colors.append(au_colors_dict[au])
        else:
            colors.append('white')
    rects.append(ax.bar(x+offset, rect, width/longest_au_len, color=colors))
    offset+=width/longest_au_len
    
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('AU Intensity')
title_emotion = ''
emotion = int(sys.argv[2])
if emotion == 0:
    title_emotion = 'Happiness'
elif emotion == 1:
        title_emotion = 'Fear'
elif emotion == 2:
        title_emotion = 'Disgust'
elif emotion == 3:
        title_emotion = 'Anger'
elif emotion == 4:
        title_emotion = 'Surprise'
elif emotion == 5:
        title_emotion = 'Sadness'
method = int(sys.argv[3]) 
if method == 0:
    ax.set_title(title_emotion+' Cluster AU Analysis (K-Means)')
else:
    ax.set_title(title_emotion+' Cluster AU Analysis (GMM)')




flat_std_list = []
for sublist in std_bars:
    for item in sublist:
        if item == 0:
            continue
        flat_std_list.append(item)
i = 0
for rect in rects:
    for bar in rect:
        height = bar.get_height()
        if height == 0:
            continue
        error_rect = ax.errorbar(bar.get_x() + bar.get_width()/2, height, yerr=flat_std_list[i], color="black", capsize=3, lw=0.5)
        ax.annotate('{}'.format(au_labels[i]),
            xy=(bar.get_x() + bar.get_width() / 2, height+flat_std_list[i]),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', size=14)
        i+=1



fig.set_size_inches(12,8)
fig.tight_layout()

plt.show()


        




