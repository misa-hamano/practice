from sklearn import datasets
from matplotlib import pyplot as plt
# from sklearn import datasets

digits = datasets.load_digits()

# 画像の配列データ
print(digits.data)

# ラベル
print(digits.target)

# 画像の表示
# number 0

plt.subplot(141), plt.imshow(digits.images[0],cmap = "gray")
plt.title("number 0"), plt.xticks([]), plt.yticks([])

# number 1
plt.subplot(142), plt.imshow(digits.images[1],cmap = "gray")
plt.title("number 1"), plt.xticks([]), plt.yticks([])

# number 2
plt.subplot(143), plt.imshow(digits.images[2],cmap =  "gray")
plt.title("number 2"), plt.xticks([]), plt.yticks([])

# number 9
plt.subplot(144), plt.imshow(digits.images[-2],cmap = "gray")
plt.title("number 9"), plt.xticks([]), plt.yticks([])

plt.show()

from sklearn import svm

SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=0.001, kernel="rbf",
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

#  SVM(識別機の事)
clf = svm.SVC(gamma = 0.001, C = 100.)
clf.fit(digits.data[:-10], digits.target[:-10])


clf.predict(digits.data[-10:])
array([5,4,8,8,4,9,0,8,9,8])

print(digits.target[-10:])