import os
import numpy as np
import pandas as pd
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

huge_number = 9999  # for initialization of DP calculation on dtw

# loading data
reference_folder1 = os.listdir('level4/reference/1')
reference_file1 = []
ref = []
for i in range(len(reference_folder1)):
    reference_file1.append(np.loadtxt(
        'level4/reference/1/'+reference_folder1[i]))
    ref.append(np.loadtxt('level4/reference/1/'+reference_folder1[i]))
reference_folder2 = os.listdir('level4/reference/2')
reference_file2 = []
for i in range(len(reference_folder2)):
    reference_file2.append(np.loadtxt(
        'level4/reference/2/'+reference_folder2[i]))
    ref.append(np.loadtxt('level4/reference/2/'+reference_folder2[i]))

test_folder = np.sort(os.listdir('level4/test'))
test_file = []
for i in range(len(test_folder)):
    test_file.append(np.loadtxt('level4/test/'+test_folder[i]))

# with tslearn library
clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
clf.fit(ref, [0, 0, 0, 1, 1, 1])
tslearn_pred = clf.predict(test_file)
clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
clf.fit(ref, [0, 0, 0, 1, 1, 1])
tslearn_pred3 = clf.predict(test_file)


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


# to print out the result table
d = [[0 for i in range(10)] for i in range(len(test_file))]
for i in range(len(reference_file1)):
    for j in range(len(test_file)):
        d[j][i] = (dtw_distance(reference_file1[i], test_file[j]))
for i in range(len(reference_file2)):
    for j in range(len(test_file)):
        d[j][i+3] = (dtw_distance(reference_file2[i], test_file[j]))
for i in range(len(test_file)):
    nearest1 = min(d[i][0:3])
    nearest2 = min(d[i][3:6])
    d[i][6] = 0 if nearest1 <= nearest2 else 1
    #
    three_nn1_1 = min(d[i][0:3])
    three_nn1_3 = max(d[i][0:3])
    three_nn1_2 = sum(d[i][0:3])-three_nn1_1-three_nn1_3
    three_nn2_1 = min(d[i][3:6])
    three_nn2_3 = max(d[i][3:6])
    three_nn2_2 = sum(d[i][3:6])-three_nn2_1-three_nn2_3
    if three_nn1_1 <= three_nn2_1:
        if three_nn1_2 <= three_nn2_1:
            d[i][7] = 0
        elif three_nn1_2 <= three_nn2_2:
            d[i][7] = 0
        else:
            d[i][7] = 1
    else:
        if three_nn1_1 >= three_nn2_2:
            d[i][7] = 1
        elif three_nn1_2 >= three_nn2_2:
            d[i][7] = 1
        else:
            d[i][7] = 0
    d[i][8] = tslearn_pred[i]
    d[i][9] = tslearn_pred3[i]

df = pd.DataFrame(d, columns=["ref1_1", "ref1_2", "ref1_3", "ref2_1", "ref2_2",
                  "ref2_3", "1-nn", "3-nn", "tslearn_n=1", "tslearn_n=3"], index=test_folder)
print(df)
