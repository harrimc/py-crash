import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

wine_file_path = 'data/winequality-red.csv'
wineframe = pd.read_csv(wine_file_path)

desired = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']
X = wineframe[desired]
y = wineframe.quality
train_X, val_X, train_y, val_y =train_test_split(X,y,random_state=2)

Forrestmodel = RandomForestRegressor(random_state=2)
Forrestmodel.fit(train_X,train_y)
prediction = Forrestmodel.predict(val_X)
mae = mean_absolute_error(prediction,val_y)
print(mae)