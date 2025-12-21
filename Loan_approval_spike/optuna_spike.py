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
import optuna 

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

model = XGBClassifier(learning_rate = 0.05, n_estimators =700,max_depth = 5,subsample=0.95, colsample_bytree=0.5,min_child_weight = 3,random_state =1, gamma = 0.029)
model1 = XGBClassifier(learning_rate = 0.05, n_estimators =700,max_depth = 5,subsample=0.95, colsample_bytree=0.5,min_child_weight = 3,random_state =0, gamma = 0.029)
model2 = XGBClassifier(learning_rate = 0.05, n_estimators =700,max_depth = 5,subsample=0.95, colsample_bytree=0.5,min_child_weight = 3,random_state =2,gamma = 0.029)
model3 = XGBClassifier(learning_rate = 0.05, n_estimators =700,max_depth = 5,subsample=0.95, colsample_bytree=0.5,min_child_weight = 3,random_state =3,gamma = 0.029)
model4 = XGBClassifier(learning_rate = 0.05, n_estimators =700,max_depth = 5,subsample=0.95, colsample_bytree=0.5,min_child_weight = 3,random_state =5,gamma = 0.029)

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

'''scores = cross_val_score(full_mod,X,y,cv =cV,scoring = 'roc_auc')

print(scores.mean())

print(scores.std())


diff_mod = [full_mod,full_mod1,full_mod2,full_mod3,full_mod4]
n = len(X)
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

print(oof_pred)

print(roc_auc_score(y, oof_pred))


def testing(pipeline,X,y,cl) : 
    mena= []
    sdt = []
    for k in [0.55,0.6,0.65,0.7,0.75] :
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
testing_df = pd.DataFrame({'no of k' : [0.55,0.6,0.65,0.7,0.75], 'mean' : mean, 'standard deviation' : std})
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

submission.to_csv('submitXG01.4.csv', index = False) '''

def objective(trial) : 
    parameters = {'learning_rate' : trial.suggest_float('learning_rate',0.01,0.5),
                  'n_estimators' : trial.suggest_int('n_estimators',400,1000),
                  'max_depth' : trial.suggest_int('max_depth',2,9),
                  'subsample' : trial.suggest_float('subsample',0.4,1),
                  'colsample_bytree' : trial.suggest_float('colsample_bytree',0.5,1),
                  'min_child_weight' : trial.suggest_int('min_child_weight',1,10),
                  'gamma' : trial.suggest_float('gamma',0,3)}
    score_op = []
    for fold, (train_idx, val_idx) in enumerate(cV.split(X,y)) :
        X_train, Val_X = X.iloc[train_idx], X.iloc[val_idx]
        y_train, Val_y = y.iloc[train_idx], y.iloc[val_idx]
        for k in [0,1,2,3,4,5,6,7,8,9,10] :
            model_op = XGBClassifier(**parameters, random_state =k)
            full_mod_op = Pipeline(steps=[('preprocessor',preprocessor), 
                                  ('model', model_op) ])
            full_mod_op.fit(X_train,y_train)
            score_preds = full_mod_op.predict_proba(Val_X)[:, 1]
            score_op.append(roc_auc_score(Val_y, score_preds))

            running_mean = np.mean(score_op)
            trial.report(running_mean, step=fold)

            if trial.should_prune():
                raise optuna.TrialPruned()# this has hella errors atm 
    return float(np.mean(score_op))

study = optuna.create_study(study_name = 'optuna_test_4.0', storage="sqlite:///optuna.db", direction='maximize',load_if_exists= True)
study.optimize(objective, n_trials=1000,show_progress_bar= True, n_jobs = -1)

best_params = study.best_params
print(best_params)

## sqlite:///optuna.db for web page 