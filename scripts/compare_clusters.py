import sys

input1 = sys.argv[1]
input2 = sys.argv[2]
numlist1 = list(map(int,input1.split(' ')))
numlist2 = list(map(int,input2.split(' ')))

num_clusters1 = max(numlist1)+1
num_clusters2 = max(numlist2)+1
scount = 1
km = [set() for i in range(num_clusters1)]
for num in numlist1:
    km[num].add('S'+str(scount))
    scount+=1

scount = 1
gmm = [set() for i in range(num_clusters2)]
for num in numlist2:
    gmm[num].add('S'+str(scount))
    scount+=1
kn = 0
for a in km:
    gmmn = 0
    for b in gmm:
        print(len(a.intersection(b)))
        print(len(a.union(b)))
        print('C'+str(kn)+' & C'+str(gmmn)+': '+str(len(a.intersection(b))/len(a.union(b))*100))
        gmmn+=1
    kn+=1
