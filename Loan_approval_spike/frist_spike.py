import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np 
from sklearn.metrics import roc_auc_score
from sklearn.base import clone


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

#defining models with different seeds
xgb_params = {'learning_rate': 0.034437561078606294, 'n_estimators': 1379, 'max_depth': 5, 'subsample': 0.939633605230379, 'colsample_bytree': 0.5379887024016051, 'min_child_weight': 4, 'gamma': 0.41725949046109445, 'reg_alpha': 0.8870451366409303, 'reg_lambda': 1.2125409014625224}
model = XGBClassifier(**xgb_params, random_state = 0)
model1 = XGBClassifier(**xgb_params, random_state = 1)
model2 = XGBClassifier(**xgb_params, random_state =2)
model3 = XGBClassifier(**xgb_params, random_state = 3)
model4 = XGBClassifier(**xgb_params, random_state = 4)

# full pipeline for each model
full_mod = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model)])
full_mod1 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model1)])
full_mod2 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model2)])
full_mod3 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model3)])
full_mod4 = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model',model4)])

cV = StratifiedKFold(n_splits =5, shuffle = True,random_state=1)

scores = cross_val_score(full_mod,X,y,cv =cV,scoring = 'roc_auc')

print(scores.mean())

print(scores.std())




diff_mod = [full_mod,full_mod1,full_mod2,full_mod3,full_mod4]
n = len(X)
fold_sc = []
oof_pred = np.zeros(n, dtype=float)  # enmpty array so can be fillef with probabilities
for train_idx, val_idx in cV.split(X, y):       ## loops five times as n_splits =5 , each time creaing differnt lists of training and val index's which constiute 5 folds
    X_train =  X.iloc[train_idx]
    y_train =  y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    sum_preds = np.zeros(len(val_idx))       # ok so this needs to be same length as val_idx as we want to add it to oof_preds[val_idx], and we want it to reset each time we loop over to another fold

    for k in diff_mod :
        k_fold = clone(k)
        k_fold.fit(X_train,y_train)          # fit on ur training data, but fit a clone to avoid permenantly altering the pipleline 
        val_preds = k_fold.predict_proba(X_val)[:, 1]
        sum_preds = sum_preds + val_preds
    
    oof_pred[val_idx]  = (sum_preds/5) 
    val_score = roc_auc_score(y_val,sum_preds/5)
    fold_sc.append(val_score)

    


print(oof_pred)
print(np.std(fold_sc))
print(roc_auc_score(y, oof_pred))


def testing(pipeline,X,y,cl) : 
    mena= []
    sdt = []
    for k in [0.939633605230379] :
        full_modtest_k = pipeline.set_params(model__subsample = k)
        score = cross_val_score(full_modtest_k,X,y,cv= cl,scoring= 'roc_auc')
        mena.append(score.mean())
        sdt.append(score.std())
    return mena,sdt

full_mod.fit(X,y)
full_mod1.fit(X,y)
full_mod2.fit(X,y)
full_mod3.fit(X,y)
full_mod4.fit(X,y)



mean , std = testing(full_mod,X,y,cV)
testing_df = pd.DataFrame({'no of k' : [0.939633605230379], 'mean' : mean, 'standard deviation' : std})
print(testing_df)



test_path = 'test.csv'
test_df = pd.read_csv(test_path)

X_valk = test_df[full_col]

preds = full_mod.predict_proba(X_valk)[:, 1]
preds1 = full_mod1.predict_proba(X_valk)[:, 1]
preds2 = full_mod2.predict_proba(X_valk)[:, 1]
preds3 = full_mod3.predict_proba(X_valk)[:, 1]
preds4 = full_mod4.predict_proba(X_valk)[:, 1]

p_avg = (preds + preds1 + preds2 + preds3 + preds4)/5

submission = pd.DataFrame({'id' : test_df['id'], 'loan_status' : p_avg})

submission.to_csv('submitXG01.11.csv', index = False)

submission2 = pd.DataFrame({'id' : test_df['id'], 'loan_status' : preds})
submission2.to_csv('submitXG01.12.csv', index = False)



