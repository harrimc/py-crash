import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

ins_path = 'train.csv'
ins_full_df = pd.read_csv(ins_path)



num_cols = ((ins_full_df.dtypes == 'int64')| (ins_full_df.dtypes == 'float64'))
num_list = list(num_cols[num_cols].index)
num_drop = ['id','loan_status']
num_final = [k for k in num_list if k not in num_drop]

cat_cols = ins_full_df.dtypes == 'object'
cat_make = list(cat_cols[cat_cols].index)
cat_list = [i for i in cat_make if i != 'loan_status']


num_transformer = SimpleImputer(strategy='mean')

cat_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy = 'most_frequent')),
                                  ('OHE', OneHotEncoder(handle_unknown = 'ignore'))])

full_col = num_final + cat_list
X = ins_full_df[full_col]
y = ins_full_df.loan_status


preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_final), 
                                               ( 'cat', cat_transformer, cat_list)])

model = XGBClassifier(learning_rate = 0.01, n_estimators =3500,max_depth = 5,subsample=0.95, colsample_bytree=0.7,min_child_weight = 3,random_state =1)

full_mod = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model)])
full_modtest = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model)])


cV = StratifiedKFold(n_splits =5, shuffle = True,random_state=1)

scores = cross_val_score(full_mod,X,y,cv =cV,scoring = 'roc_auc')

print(scores.mean())

fit_model = full_mod.fit(X,y)

print(scores.std())


def testing(pipeline,X,y,cl) : 
    mena= []
    sdt = []
    for k in [0.65,0.7,0.72,0.75, 0.8] :
        full_modtest_k = pipeline.set_params(model__colsample_bytree = k)
        score = cross_val_score(full_modtest_k,X,y,cv= cl,scoring= 'roc_auc')
        mena.append(score.mean())
        sdt.append(score.std())
    return mena,sdt



mean , std = testing(full_modtest,X,y,cV)
testing_df = pd.DataFrame({'no of k' : [0.65,0.7,0.72,0.75, 0.8], 'mean' : mean, 'standard deviation' : std})
print(testing_df)
























test_path = 'test.csv'
test_df = pd.read_csv(test_path)

X_val = test_df[full_col]

preds = fit_model.predict_proba(X_val)[:, 1]

submission = pd.DataFrame({'id' : test_df['id'], 'loan_status' : preds})

submission.to_csv('submitXG01.2.csv', index = False)
