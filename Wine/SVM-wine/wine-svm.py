import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Header
text = ' WINE - SVM CLASSIFICATION '
print('\033[1;30m', text.center(40, '#'), '\033[1;m')

# Readin Dataset
df = pd.read_csv('wine.csv')

# Creating Train and Test Data
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)
print("Train Size:{} || Test Size:{}" .format(np.size(y_train), np.size(y_test)))

# SVM Classification
# Tuned Parameters
gammas = [1e-2, 1e-3, 1e-4, 1e-5]
Cs = [1, 10, 100, 1000]
train_accuracy = []
test_accuracy = []
parameters = []

for gamma in gammas:
    for c in Cs:
        clf = svm.SVC(kernel='poly', decision_function_shape='ovo', gamma=gamma, C=c)
        clf.fit(X_train, y_train)
        train_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))
        parameters.append([gamma, c])

# Best Train Accuracy
best_train_accuracy = round(np.max(train_accuracy)*100, 3)
best_train_params = parameters[train_accuracy.index(np.max(train_accuracy))]
print('\033[1;33mBest Train Accuracy: {}% params:{} \033[1;m' .format(best_train_accuracy, best_train_params))

# Test Accuracy With Best Train Accuracy Params
print('Params:{} (For Best Train) Test Accuracy: {}% \033[1;m' .format(best_train_params,
      round(test_accuracy[train_accuracy.index(np.max(train_accuracy))]*100, 3)))

# Best Test Accuracy
best_test_accuracy = round(np.max(test_accuracy)*100, 3)
best_test_params = parameters[test_accuracy.index(np.max(test_accuracy))]
print('\033[1;32mBest Test Accuracy: {}% params:{} \033[1;m' .format(best_test_accuracy, best_test_params))

# Prediction for Best Accuracy
BestClf = svm.SVC(kernel='poly', decision_function_shape='ovo', gamma=best_test_params[0], C=best_test_params[1])
BestClf.fit(X_train, y_train)
prediction = BestClf.predict(X_test)

# Kappa Score
print("Kappa Score:{}" .format(round(cohen_kappa_score(y_test, prediction), 3)))

# Classification Report
print("\n\033[1;30m", classification_report(y_test, prediction), "\033[1;m")

# AUC Score
def multiclass_roc_auc_score(ytest, ypred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(ytest)
    ytest = lb.transform(ytest)
    ypred = lb.transform(ypred)
    return roc_auc_score(ytest, ypred, average=average)

print('AUC:{}' .format(round(multiclass_roc_auc_score(y_test, prediction), 2)))

# 3D Plotting
fig = plt.figure(1, figsize=(10, 6))
ax = Axes3D(fig)
colors = {1: 'r', 2: 'g', 3: 'b'}

# Plot Training Data
# ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=[colors[i] for i in y_train],
#            s=40, marker='o', label='Train Data')

# Plot Test Data
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=[colors[i] for i in prediction],
           s=40, marker='*', label='Test Data')

# Showing False prediction
n = np.size(prediction)
for i in range(0, n):
    if prediction[i] != y_test[i]:
        ax.scatter(X_test[i, 0], X_test[i, 1], X_test[i, 2], c=colors[y_test[i]], marker='X', s=80)


# Graph information
ax.set_title('SVM Classification for Wine Data Set')
ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('Ash')
legend_elements = [Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor='r', markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='2', markerfacecolor='g', markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='3', markerfacecolor='b', markersize=7),
                   # Line2D([0], [0], marker='o', color='w', label='Train Data', markerfacecolor='k', markersize=7),
                   Line2D([0], [0], marker='*', color='w', label='Test Data', markerfacecolor='k', markersize=10),
                   Line2D([0], [0], marker='X', color='w', label='Correct Class', markerfacecolor='k', markersize=10)]
ax.legend(handles=legend_elements)
plt.show()