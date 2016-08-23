import pandas as pd
import numpy as np 
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

iris_df = pd.DataFrame(X, columns = iris.feature_names[2:])

print iris_df.head()