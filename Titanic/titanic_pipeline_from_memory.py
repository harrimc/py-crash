import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold


train_path = 'train.csv'
full_data = pd.read_csv(train_path)


num_mask = ((full_data.dtypes == 'int64') | (full_data.dtypes == 'float64'))
num_col = list(num_mask[num_mask].index)
num_drop = ['PassengerId','Survived']
num_final = [k for k in num_col if k not in num_drop]


cat_mask = (full_data.dtypes == 'object')
cat_list = list(cat_mask[cat_mask].index)
cat_drop = ['Name','Ticket', 'Cabin']
cat_final = [i for i in cat_list if i not in cat_drop]

num_transformer = SimpleImputer(strategy='mean')

cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('encoding', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num',num_transformer,num_final),
                                               ('cat', cat_transformer,cat_final)])

y = full_data.Survived

useable_cols = num_final + cat_final
X = full_data[useable_cols]

model = XGBClassifier(learning_rate = 0.05, n_estimators =300,max_depth = 3,subsample=0.8, colsample_bytree=0.8,min_child_weight = 5,random_state =1 )

full_mod = Pipeline(steps=[('preprocessor',preprocessor),
                           ('model',model)])

cV = StratifiedKFold(n_splits =4, shuffle=True,random_state=1)


scores= cross_val_score(full_mod,X,y,cv = cV, scoring = 'accuracy')

print(scores.mean())

fit_model = full_mod.fit(X,y)

'------------------------------------------------------------------'

test_path = 'test.csv'
test_pd = pd.read_csv(test_path)

X_val = test_pd[useable_cols]

preds = full_mod.predict(X_val)

pre_df = pd.DataFrame({'PassengerId' : test_pd['PassengerId'], 'Survived' : preds})

pre_df.to_csv('Submission4CGBMemory.csv', index=False)