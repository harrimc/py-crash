import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import sklearn.ensemble as se



wine_file_path = 'data/winequality-red.csv'

wineframe = pd.read_csv(wine_file_path)
#print(wineframe.head)
#print(wineframe.columns)

desired = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']
X = wineframe[desired]
y = wineframe.quality

train_X, val_X, train_y, val_y =train_test_split(X,y,random_state=1)
#print(train_X.head)


wine_tree = DecisionTreeRegressor(random_state=1)
wine_tree.fit(train_X,train_y)
guess = wine_tree.predict(val_X)

actual = mean_absolute_error(guess, val_y)
print(actual)
