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

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state = 1)

def get_rmsle(max_features,max_leaf_nodes,train_X,val_x,train_y,val_y) :
    houseforrest = RandomForestRegressor(max_features=max_features, max_leaf_nodes=max_leaf_nodes,random_state=1)
    houseforrest.fit(train_X,train_y)
    predict = houseforrest.predict(val_X)
    rmsle = root_mean_squared_log_error(val_y,predict)
    return rmsle


for k in [700] :
    print(get_rmsle(7,k,train_X,val_X,train_y,val_y))


