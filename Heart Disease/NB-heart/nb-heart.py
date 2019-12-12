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

text = ' HEART DISEASE - NAIVE BAYES CLASSIFICATION '
print('\033[1;30m', text.center(40, '#'), '\033[1;m')

df = pd.read_csv('heart.csv')

X = np.array(df.drop(['target'], 1))
y = np.array(df['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)
print('Train Size:{} || Test Size:{}' .format(np.size(y_train), np.size(y_test)))

# Naive Bayes Classification
clf = GaussianNB(priors=None)
clf.fit(X_train, y_train)


prediction = clf.predict(X_test)

# Train and Test Accuracy
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

print('\033[1;33mTrain Accuracy: {}%\033[1;m' .format(round(train_accuracy*100, 3)))
print('\033[1;32mTest Accuracy: {}%\033[1;m' .format(round(test_accuracy*100, 2)))

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


ax.set_title("Naive Bayes Classification for Heart Disease Dataset")
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