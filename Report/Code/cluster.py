# -*- coding: utf-8 -*-
"""
VGIS Group 843
Rita and Atanas
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import re
from sklearn import svm


def do_stuff(data, name):
    file = open(name+'.txt', 'r')
    for line in file:
        line_data = line.split(' ')
        for i in range(len(line_data)):
            line_data[i] = int(re.sub("[^0-9]", "", line_data[i]))
        data.append(line_data)

    file.close()
    return data


def classify(svm, scaler, x, y):
    sample = scaler.transform([x, y])
    result = svm.predict(sample)
    return result


def get_lables(lab):
    for i in range(len(lab)):
        if lab[i] != 0:
            lab[i] = 1
            print(lab[i])
    return lab


data = []

data = do_stuff(data, 'testfile')
data = do_stuff(data, 'testfile1')
data = do_stuff(data, 'testfile2')
data = do_stuff(data, 'testfile3')
data = do_stuff(data, 'testfile4')
data = do_stuff(data, 'testfile5')
data = do_stuff(data, 'testfile6')
data = do_stuff(data, 'testfile7')


data = np.asarray(data)[:, 0:2]

np.savetxt('data.txt', data)
scaler = StandardScaler().fit(data)
data = scaler.transform(data)


db = DBSCAN(eps=0.1, min_samples=10).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    # Core points
    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    # Border points
    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

labeled_data = np.zeros([len(data), len(data[0])+1])
labels_binary = np.zeros(len(labels))
print(labels_binary.shape)
for i in range(len(data)):
    labeled_data[i, 0:len(data[i])] = data[i]
    labeled_data[i, len(data[i]+1)] = labels[i]
    if labels[i] != 0:
        labels_binary[i] = 1
print(labels_binary.shape)

np.savetxt('train_labels.txt', labels_binary)
clf = svm.SVC()
clf.fit(data[0:7000], labels_binary[0:7000])

pred = clf.predict(data[7000:])

print(sum(labels_binary[7000:]-pred)/len(pred))

