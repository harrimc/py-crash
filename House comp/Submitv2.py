import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split


traindata_path = 'train.csv'
housedata = pd.read_csv(traindata_path)

numeric_cols = housedata.select_dtypes(include =['int64','float64'])

cols = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold', ]

X = housedata[cols]
y = housedata.SalePrice

train_X,val_X,train_y,val_y = train_test_split(X,y,train_size = 0.86, test_size= 0.14,random_state = 1)

final_model = RandomForestRegressor(max_features=7, max_leaf_nodes=540,random_state=1)
final_model.fit(X,y)

test_path = 'test.csv'
test_data = pd.read_csv(test_path)

val_X = test_data[cols]

test_preds = final_model.predict(val_X)

susbmission = pd.DataFrame({'Id' : test_data['Id'], 'SalePrice' : test_preds})

susbmission.to_csv('CREATE CSV',index = False)