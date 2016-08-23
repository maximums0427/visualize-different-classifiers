import pandas as pd
import numpy as np 
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

iris_df = pd.DataFrame(X, columns = iris.feature_names[2:])

print iris_df.head()
print 'This dataset has {} unique labels.\n'.format(np.unique(y))

#split data to training and testing sets
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
													random_state = 0)

print 'There are {} samples in the training set and {} samples in the test set.\n'.format(X_train.shape[0], X_test.shape[0])

#normaliza data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print pd.DataFrame(X_train_std,columns = iris_df.columns).head()


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

markers = ('s', 'x', 'o')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
	plt.scatter(x = X[y==cl, 0], y = X[y==cl,1],
				c = cmap(idx), marker = markers[idx], label = cl)
plt.show()





