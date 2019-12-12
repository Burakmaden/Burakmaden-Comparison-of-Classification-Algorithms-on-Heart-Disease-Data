import matplotlib.pyplot as plt
import numpy as np

#       SVM     KNN     NB    DT
svm = [100.0, 97.22, 70.93, 87.71]

knn = [93.33, 76.39, 70.93, 68.03]

nb = [98.33, 100.0, 32.56, 81.97]

dt = [100.0, 97.22, 73.26, 84.43]

N = 4
ind = np.arange(N)
width = 0.08
plt.bar(ind, svm, width, label='SVM', color='b', alpha=.75)
plt.bar(ind + width, knn, width, label='k-NN', color='r', alpha=.75)
plt.bar(ind + 2*width, nb, width, label='Naive Bayes', color='c', alpha=.75)
plt.bar(ind + 3*width, dt, width, label='Decision Tree', color='g', alpha=.75)

plt.ylabel('Accuracies')
plt.title('Classification Accuracy Results in Graph')

plt.xticks(ind + width, ('Iris', 'Wine', 'Glass', 'Heart Disease'))
plt.legend(loc='best')
plt.show()