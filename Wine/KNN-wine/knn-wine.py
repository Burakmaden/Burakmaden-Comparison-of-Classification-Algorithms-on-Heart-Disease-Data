import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Header
text = ' WINE - KNN CLASSIFICATION '
print('\033[1;30m', text.center(40, '#'), '\033[1;m')

df = pd.read_csv('wine.csv')

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)

neigh = np.arange(3, 11)
train_accuracy = []
test_accuracy = []

# K-NN Classification
for i, k in enumerate(neigh):
    # p=2 | Eucledian Distance
    # weights='uniform' | All points in each neighborhood are weighted equally
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, p=2, weights='uniform')
    clf.fit(X_train, y_train)
    train_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))


# Best Train Accuracy
best_train_accuracy = round(np.max(train_accuracy)*100, 3)
best_train_accuracy_K = train_accuracy.index(np.max(train_accuracy)) + 3
print('\033[1;33mBest Train Accuracy: {}% K:{} \033[1;m' .format(best_train_accuracy, best_train_accuracy_K))

# Test Accuracy for Best Train Parameters
print('K:{} (For Best Train) Test Accuracy: {}%'
      .format(best_train_accuracy_K, round(test_accuracy[train_accuracy.index(np.max(train_accuracy))]*100, 3)))

# Best Test Accuracy
best_test_accuracy = round(np.max(test_accuracy)*100, 3)
best_test_accuracy_K = test_accuracy.index(np.max(test_accuracy)) + 3
print('\033[1;32mBest Test Accuracy: {}% K:{} \033[1;m' .format(best_test_accuracy, best_test_accuracy_K))

# Prediction for Best Accuracy
Best_K = best_test_accuracy_K
BestClf = neighbors.KNeighborsClassifier(n_neighbors=Best_K, p=2, weights='uniform')
BestClf.fit(X_train, y_train)
prediction = BestClf.predict(X_test)

# Kappa Score
print("Kappa Score:{}" .format(round(cohen_kappa_score(y_test, prediction), 3)))

# AUC Score
def multiclass_roc_auc_score(ytest, ypred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(ytest)
    ytest = lb.transform(ytest)
    ypred = lb.transform(ypred)
    return roc_auc_score(ytest, ypred, average=average)

print('AUC:{}' .format(round(multiclass_roc_auc_score(y_test, prediction), 2)))

# Classification Report
print("\n\033[1;30m", classification_report(y_test, prediction), "\033[1;m")

# 3D Plotting
fig = plt.figure(figsize=(10, 6))
ax = Axes3D(fig)
colors = {1: 'r', 2: 'g', 3: 'b'}

# Train Data Plot
# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=[colors[i] for i in y_train],
#            s=40, marker='o', label='Train Data')

# Test Data Plot
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=[colors[i] for i in prediction],
           s=40, marker='*', label='Train Data')

# Showing False Prediction
n = np.size(prediction)
for i in range(0, n):
    if prediction[i] != y_test[i]:
        ax.scatter(X_test[i, 0], X_test[i, 1], X_test[i, 2], c=colors[y_test[i]], alpha=.8, marker='X', s=80)

# Graph info
ax.set_title("K-NN Classification for Wine Data set")
ax.set_xlabel("Alcohol")
ax.set_ylabel("Malic Acid")
ax.set_zlabel("Ash")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor='r', markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='2', markerfacecolor='g', markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='3', markerfacecolor='b', markersize=7),
                   # Line2D([0], [0], marker='o', color='w', label='Train Data', markerfacecolor='k', markersize=7),
                   Line2D([0], [0], marker='*', color='w', label='Test Data', markerfacecolor='k', markersize=10),
                   Line2D([0], [0], marker='X', color='w', label='Correct Class', markerfacecolor='k', markersize=10)]
ax.legend(handles=legend_elements)
plt.show()