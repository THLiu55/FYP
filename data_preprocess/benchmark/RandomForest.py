# kNN with neighbors=4 benchmark for Kuzushiji-MNIST
# Acheives 92.10% test accuracy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

def load(f):
    return np.load(f)['arr_0']

# Load the data
x_train = load('../k49-train-imgs.npz')
x_test = load('../k49-test-imgs.npz')
y_train = load('../k49-train-labels.npz')
y_test = load('../k49-test-labels.npz')

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
clf = RandomForestClassifier(n_estimators=100, max_depth=30, max_features=28)

print('Fitting', clf)

start_time = time.time()

clf.fit(x_train, y_train)

end_time = time.time()
execution_time = end_time - start_time
print(f"代码执行时间：{execution_time} 秒")
print('Evaluating', clf)

test_score = clf.score(x_test, y_test)
print('Test accuracy:', test_score)
