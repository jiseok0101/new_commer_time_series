import numpy as np
import os
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

huge_number = 9999

reference_1 = np.loadtxt("level3/reference/1.dat")
reference_2 = np.loadtxt("level3/reference/2.dat")

test_folder = np.sort(os.listdir('level3/test'))

test_file = []
for i in range(len(test_folder)):
    test_file.append(np.loadtxt('level3/test/'+test_folder[i]))

# with tslearn library
ref = []
ref.append(np.loadtxt("level3/reference/1.dat"))
ref.append(np.loadtxt("level3/reference/2.dat"))
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
