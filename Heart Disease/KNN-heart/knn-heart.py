import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report

# Header
text = 'HEART DISEASE - KNN CLASSIFICATION '
print('\033[1;30m', text.center(40, '#'), '\033[1;m')

# Reading Dataset
df = pd.read_csv('heart.csv')

# Creating Train and Test Data
X = np.array(df.drop(['target'], 1))
y = np.array(df['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)
print('Train Data Size:', np.size(y_train), '// Test Data Size:', np.size(y_test))

# K-NN Classification
neigh = np.arange(3, 11)
train_accuracy = []
test_accuracy = []

for i, k in enumerate(neigh):
    # p=2 | Eucledian Distance
    # weights='uniform' | All points in each neighborhood are weighted equally
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, p=2, weights='uniform')
    clf.fit(X_train, y_train)
    train_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append((clf.score(X_test, y_test)))


# Best Train Accuracy
best_train_accuracy = round(np.max(train_accuracy)*100, 3)
best_train_accuracy_K = train_accuracy.index(np.max(train_accuracy)) + 3
print('\033[1;33mBest Train Accuracy: {}% K:{} \033[1;m' .format(best_train_accuracy, best_train_accuracy_K))

# Test Accuracy With Best Train Accuracy Neighbors
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
print("Kappa Score:{}" .format(cohen_kappa_score(y_test, prediction)))

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

# 3D plotting
fig = plt.figure(1, figsize=(10, 6))
ax = Axes3D(fig)
colors = {0: 'b', 1: 'r'}

# Training Data Plot scatter
# ax.scatter(X_train[:, 0], X_train[:, 3], X_train[:, 7], c=[colors[i] for i in y_train],
#            s=40, marker='o', label='Train Data')

# Test Data Plot Scatter
ax.scatter(X_test[:, 0], X_test[:, 3], X_test[:, 7], c=[colors[i] for i in prediction],
           s=40, marker='*', label='Test Data')

# Showing False prediction
n = np.size(prediction)
for i in range(0, n):
    if prediction[i] != y_test[i]:
        ax.scatter(X_test[i, 0], X_test[i, 3], X_test[i, 7], c=colors[y_test[i]], marker='X', s=80)


ax.set_title("K-NN Classification for Heart Disease Data set")
ax.set_xlabel("age")
ax.set_ylabel("trestbps")
ax.set_zlabel("thalach")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='0', markerfacecolor=colors[0], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor=colors[1], markersize=7),
                   # Line2D([0], [0], marker='o', color='w', label='Train Data', markerfacecolor='k', markersize=7),
                   Line2D([0], [0], marker='*', color='w', label='Test Data', markerfacecolor='k', markersize=10),
                   Line2D([0], [0], marker='X', color='w', label='Correct Class', markerfacecolor='k', markersize=10)]
ax.legend(handles=legend_elements)
plt.show()