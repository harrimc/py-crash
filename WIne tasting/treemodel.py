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


def get_tree_mae(max_leaf_nodes,train_X, val_X, train_y, val_y) :
    wine_tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    wine_tree.fit(train_X,train_y)
    guess = wine_tree.predict(val_X)
    actual = mean_absolute_error(val_y, guess)
    return actual 

print(get_tree_mae(34,train_X,val_X,train_y,val_y))



