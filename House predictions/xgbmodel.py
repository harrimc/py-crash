from xgboost import XGBRegressor
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import root_mean_squared_log_error


house_path = 'train.csv'
hous_fulldata = pd.read_csv(house_path) 


num_col = ((hous_fulldata.dtypes == 'int64') | ( hous_fulldata.dtypes == 'float64'))
ls_num_col = list(num_col[num_col].index)
useless_num_col = ['Id', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath','GarageYrBlt', '3SsnPorch', 'PoolArea', 'MiscVal','SalePrice']
clean_num_col = [col for col in ls_num_col if col not in useless_num_col]

cat_col = (hous_fulldata.dtypes == 'object')
ls_cat_col = list(cat_col[cat_col].index)
useless_cat_col = ['Street', 'Alley', 'LandContour', 'Utilities', 'LandSlope','Condition1', 'Condition2', 'RoofMatl', 'MasVnrType','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','Heating', 'Electrical', 'Functional', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','PoolQC', 'Fence', 'MiscFeature', 'SaleType']
clean_cat_col = [ cat for cat in ls_cat_col if cat not in useless_cat_col]


numerical_transformer = SimpleImputer(strategy='mean')

catagorical_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')), 
                                           ('encoding', OneHotEncoder(handle_unknown='ignore'))])

combined_cols = clean_num_col + clean_cat_col
X= hous_fulldata[combined_cols]
y = hous_fulldata.SalePrice

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, clean_num_col),
                                               ('cat', catagorical_transformer,clean_cat_col)])

model = XGBRegressor(n_estimators = 900, learning_rate = 0.05, max_depth = 3)

my_model = Pipeline(steps=[('preprocessor', 
                            preprocessor),('model',model)])

cV = KFold(n_splits = 5, shuffle= True, random_state = 1)

score = -1 * cross_val_score(my_model,X,y, cv = cV, scoring = 'neg_root_mean_squared_log_error')
print(score.mean())

my_model.fit(X,y)

## submission

test_path = 'test.csv'
test_pd = pd.read_csv(test_path)
X_val = test_pd[combined_cols]
preds = my_model.predict(X_val)
submission = pd.DataFrame({'Id': test_pd['Id'], 'SalePrice' : preds})

submission.to_csv('submitwithcgb.csv', index = False)


