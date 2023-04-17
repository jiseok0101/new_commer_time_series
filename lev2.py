import numpy as np
import os
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

huge_number = 9999

reference_1 = np.loadtxt("level2/reference/1.dat")
reference_2 = np.loadtxt("level2/reference/2.dat")

test_folder = np.sort(os.listdir('level2/test'))
test_file = []
for i in range(len(test_folder)):
    test_file.append(np.loadtxt('level2/test/'+test_folder[i]))

# with tslearn library
ref = []
ref.append(reference_1)
ref.append(reference_2)
# 1-nearest neighbors
clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
clf.fit(ref, [0, 1])
tslearn_pred = clf.predict(test_file)

# full scratch
# np.linalg.norm(A - B)   :vector distance


def dtw_distance(X, Y):  # X <- test data, Y <- standard data
    abs_d = [[0 for i in range(len(Y))] for i in range(len(X))]
    dtw_d = [[0 for i in range(len(Y)+1)] for i in range(len(X)+1)]
    for i in range(len(X)):
        for j in range(len(Y)):
            abs_d[i][j] = np.linalg.norm(X[i]-Y[j])
    for i in range(len(X)+1):
        for j in range(len(Y)+1):
            if i == 0:
                if j == 0:
                    dtw_d[i][j] = 0
                else:
                    dtw_d[i][j] = huge_number
            elif j == 0:
                dtw_d[i][j] = huge_number
            else:
                dtw_d[i][j] = min(dtw_d[i-1][j], dtw_d[i-1][j-1],
                                  dtw_d[i][j-1]) + pow(abs_d[i-1][j-1], 2)
    return pow(dtw_d[len(X)][len(Y)], 0.5)


d = [[0 for i in range(4)] for i in range(len(test_file))]
for i in range(len(test_file)):
    d[i][0] = dtw_distance(reference_1, test_file[i])
    d[i][1] = dtw_distance(reference_2, test_file[i])
    d[i][2] = 0 if d[i][0] <= d[i][1] else 1
    d[i][3] = tslearn_pred[i]
df = pd.DataFrame(d, columns=["ref1", "ref2",
                  "1-nn", "tslearn_1-nn"], index=test_folder)
print(df)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.plot(reference_1[:,0], reference_1[:,1], reference_2[:,2], c='red', marker='*', linestyle='-',label='reference 1')
ax.plot(reference_2[:,0], reference_2[:,1], reference_2[:,2], c='green',marker='*', linestyle='-', label='reference 2')
for i in range(len(test_file)):
    ax.plot(test_file[i][:,0], test_file[i][:,1], test_file[i][:,2], linestyle='--', linewidth='1.5', label='test '+str(i+1))
plt.legend()
plt.show()
