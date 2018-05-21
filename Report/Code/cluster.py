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


def do_stuff(data, name):
    file = open(name+'.txt', 'r')
    for line in file:
        line_data = line.split(' ')
        for i in range(3):
            line_data[i] = int(re.sub("[^0-9]", "", line_data[i]))
        data.append(line_data)

    file.close()
    return data


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
print(data[0])

data = StandardScaler().fit_transform(data)


db = DBSCAN(eps=0.095, min_samples=10).fit(data)
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

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
