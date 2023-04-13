import numpy as np
import matplotlib.pyplot as plt
import os
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import pandas as pd

huge_number = 9999  # for initialization of DP calculation on dtw

reference_1 = np.loadtxt('level1/reference/1.dat', dtype=np.float32)
reference_2 = np.loadtxt('level1/reference/2.dat', dtype=np.float32)

# graph
plt.plot(reference_1, label='reference1', linewidth=3)
plt.plot(reference_2, label='reference2', linewidth=3)
test_folder = np.sort(os.listdir('level1/test'))
test_file = []
for i in range(len(test_folder)):
    test_file.append(np.loadtxt('level1/test/'+test_folder[i]))
for i in range(len(test_file)):
    plt.plot(test_file[i], label=test_folder[i], linestyle='--')
plt.legend()
plt.show()


# full scratch
def dtw_distance(X, Y):  # X <- test data, Y <- standard data
    abs_d = [[0 for i in range(30)] for i in range(30)]
    dtw_d = [[0 for i in range(31)] for i in range(31)]
    for i in range(len(X)):
        for j in range(0, len(Y)):
            abs_d[i][j] = abs(X[i]-Y[j])  # euclidean distance
    for i in range(len(X)+1):
        for j in range(len(Y)+1):
            if i == 0:
                dtw_d[i][j] = 0 if j == 0 else huge_number
            elif j == 0:
                dtw_d[i][j] = huge_number
            else:
                dtw_d[i][j] = min(dtw_d[i-1][j], dtw_d[i-1]
                                  [j-1], dtw_d[i][j-1]) + pow(abs_d[i-1][j-1], 2)
    return pow(dtw_d[30][30], 0.5)


# with tslearn
ref = []
ref.append(reference_1)
ref.append(reference_2)

# 1-nearest neighbor
clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
clf.fit(ref, [0, 1])
tslearn_pred = clf.predict(test_file)

d = [[0 for i in range(4)] for i in range(len(test_file))]
for i in range(len(test_file)):
    d[i][0] = dtw_distance(reference_1, test_file[i])
    d[i][1] = dtw_distance(reference_2, test_file[i])
    d[i][2] = 0 if d[i][0] <= d[i][1] else 1
    d[i][3] = tslearn_pred[i]
df = pd.DataFrame(d, columns=["ref1", "ref2",
                  "1-nn", "tslearn_1-nn"], index=test_folder)
print(df)
