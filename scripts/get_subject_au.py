import sys
import numpy as np
import pandas as pd

#import data


names = ['Andre', 'Conrado', 'Diogo', 'Gabriel', 'Julia', 'Maurer', 'Paulo', 'Pedro', 'Rovane', 'Victor', 'Wetzel']


#df.reset_index(drop=True, inplace=True)
#print(df)

#create new final df
index = []
for i in range(0, 42):
    index.append('S'+str(i+1))
columns = ['Happiness', 'Fear', 'Disgust', 'Anger', 'Surprise', 'Sadness']

subs_df = pd.DataFrame(index=index, columns=columns)

sub_count = 1
for name in names:
    df = (pd.read_csv(r"D:\Users\jujum\OneDrive\Documents\IC\Emotion Style CSVs" + "\\" + name + r" All.csv",header=None, skiprows=2))
    for i in range (0, 6):
        #aus_present = []
        intensity_sum = 0.0
        emotion_df = df.iloc[[i]]
        if len(emotion_df.columns) >= 17:
            emotion_df = emotion_df.iloc[:, :-1]
        emotion_df.columns = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26"]
        for au in emotion_df.columns:
            intensity_sum += float(emotion_df[au])
            #if float(emotion_df[au]) > 0.0:
            #    aus_present.append(au)
        row_index = 'S'+str(sub_count)
        col_index = 'Happiness' if i == 0 else 'Fear' if i == 1 else 'Disgust' if i == 2 else 'Anger' if i == 3 else 'Surprise' if i == 4 else 'Sadness'
        subs_df[col_index][row_index] = intensity_sum#str(aus_present)
    sub_count+=1


df2 = pd.read_csv(r"D:\Users\jujum\OneDrive\Documents\IC\FacesDB-All.csv", header=None)
facesDB_emotions = [0, 5, 4, 3, 2, 1]
for emotion in facesDB_emotions: #emotion: 0 - happiness    1 - sadness     2 - surprise    3 - anger    4 - disgust     5 - fear
    emotion_index = emotion
    sub_count = 12
    for i in range(0,36):
        #aus_present = []
        intensity_sum = 0.0
        emotion_df = df2.iloc[[emotion_index]]
        emotion_df.columns = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26"]
        for au in emotion_df.columns:
            intensity_sum += float(emotion_df[au])
            #if float(emotion_df[au]) > 0.0:
                #aus_present.append(au)
        row_index = 'S'+str(sub_count)
        col_index = 'Happiness' if emotion == 0 else 'Fear' if emotion == 5 else 'Disgust' if emotion == 4 else 'Anger' if emotion == 3 else 'Surprise' if emotion == 2 else 'Sadness'
        subs_df[col_index][row_index] = intensity_sum#str(aus_present)
        emotion_index += 6
        sub_count+=1
print(subs_df)
subs_df.to_excel("all_subs_intensity.xlsx")
