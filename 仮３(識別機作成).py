from sklearn import svm

#  SVM(識別機の事)
clf = svm.SVC(gamma = 0.001, C = 100.0)
clf.fit(digits.data[:-10], digits.target[:-10])

SVC(C = 100.0, cache_size = 200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=0.001, kernel="rbf",
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


clf.predict(digits.data[-10:])
array([5,4,8,8,4,9,0,8,9,8])

print(digits.target[-10:])