import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Header
text = ' GLASS - NAIVE BAYES CLASSIFICATION '
print('\033[1;30m', text.center(40, '#'), '\033[1;m')

df = pd.read_csv('glass.csv')
df.drop(['Id'], 1, inplace=True)

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

# Random state 22
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=22)
print('Train Data Size:{} || Test Data Size:{}' .format(np.size(y_train), np.size(y_test)))

# Training
clf = GaussianNB(priors=None)
clf.fit(X_train, y_train)

# Prediction
prediction = clf.predict(X_test)

# Train and Test Accuracy
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print('\033[1;33mTrain Accuracy: {}%\033[1;m' .format(round(train_accuracy*100, 3)))
print('\033[1;32mTest Accuracy: {}%\033[1;m' .format(round(test_accuracy*100, 3)))

# Kappa Score
print("\n\033[1;30mKappa Score:{}\033[1;m" .format(cohen_kappa_score(y_test, prediction)))

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
fig = plt.figure(1, figsize=(10, 6))
ax = Axes3D(fig)
colors = {1: 'r', 2: 'g', 3: 'b', 4: 'k', 5: 'm', 6: 'y', 7: 'c'}

# Plot Training Data
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=[colors[i] for i in y_train],
           s=40, marker='o', label='Train Data')

# Plot Test Data
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=[colors[i] for i in prediction],
           s=40, marker='*', label='Test Data')

# Showing False prediction
n = np.size(prediction)
for i in range(0, n):
    if prediction[i] != y_test[i]:
        ax.scatter(X_test[i, 0], X_test[i, 1], X_test[i, 2], c=colors[y_test[i]], marker='X', s=80)

ax.set_title('NB Classification for Glass Data Set')
ax.set_xlabel('RefractiveIndex')
ax.set_ylabel('Sodium')
ax.set_zlabel('Magnesium')
legend_elements = [Line2D([0], [0], marker='o', color='w', label='1', markerfacecolor=colors[1], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='2', markerfacecolor=colors[2], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='3', markerfacecolor=colors[3], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='5', markerfacecolor=colors[5], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='6', markerfacecolor=colors[6], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='7', markerfacecolor=colors[7], markersize=7),
                   Line2D([0], [0], marker='o', color='w', label='Train Data', markerfacecolor='k', markersize=7),
                   Line2D([0], [0], marker='*', color='w', label='Test Data', markerfacecolor='k', markersize=10)]
ax.legend(handles=legend_elements)
# plt.show()
