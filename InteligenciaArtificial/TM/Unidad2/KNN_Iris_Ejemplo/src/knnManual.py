import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

dataTrain = [[7.4, 2.8, 6.1, 1.9],
             [7.9, 3.8, 6.4, 2],
             [6.4, 2.8, 5.6, 2.2],
             [5.7, 3, 4.2, 1.2],
             [5.7, 2.9, 4.2, 1.3],
             [6.2, 2.9, 4.3, 1.3],
             [5.1, 2.5, 3, 1.1],
             [5.4, 3.9, 1.7, 0.4],
             [4.6, 3.4, 1.4, 0.3],
             [5, 3.4, 1.5, 0.2]]
arrayDataTrain = np.array(dataTrain)

targetTrain = [2, 2, 2, 1, 1, 1, 1, 0, 0, 0]
arrayTargetTrain = np.array(targetTrain)

dataTest = [
        [5.1, 3.8, 1.5, 0.3],
        [5.4, 3.4, 1.7, 0.2],
        [6.7, 3, 5, 1.7],
        [6, 2.9, 4.5, 1.5],
        [6.7, 3, 5.2, 2.3],
        [6.3, 2.5, 5, 1.9]]
arrayDataTest = np.array(dataTest)

targetTest = [0, 0, 1, 1, 2, 2]
arrayTargetTest = np.array(targetTest)

# print(arrayData)
# print(arrayTarget)

np.random.seed(0)
indices = np.random.permutation(len(arrayDataTrain))
iris_X_train = arrayDataTrain
iris_Y_train = arrayTargetTrain
iris_X_test = arrayDataTest
iris_Y_test = arrayTargetTest

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(iris_X_train, iris_Y_train)

output = knn.predict(iris_X_test)
print("output\n", output)
print("resultados esperados\n", iris_Y_test)

# Accuracy score
accuracy = accuracy_score(iris_Y_test, output)
print("Exactitud : %0.1f%% " % (accuracy * 100))

print("Matriz de confusion\n", confusion_matrix(iris_Y_test, output))