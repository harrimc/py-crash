import pandas as pd   ### notes of continuance at the bottom 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


my_imputer = SimpleImputer()

titan_path = 'train.csv'
fulldata = pd.read_csv(titan_path)

enc = OneHotEncoder(handle_unknown= 'ignore')
encoded = enc.fit_transform(fulldata[['Sex', 'Embarked']])

encod_df = pd.DataFrame(encoded.toarray(), columns= enc.get_feature_names_out())

n_encod_df = encod_df.drop('Embarked_nan', axis=1)

X_base = fulldata.drop(['Sex','Embarked'], axis=1)

new_full_df = pd.concat([n_encod_df,X_base],axis =1)


cols = ['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',]

X = new_full_df[cols]
y = fulldata.Survived

train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=1)

imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))

imputed_train_X.columns = train_X.columns 
imputed_val_X.columns = val_X.columns   

titan_mod = RandomForestClassifier(random_state=1)                  
titan_mod.fit(imputed_train_X,train_y)
pres = titan_mod.predict(imputed_val_X)
print(accuracy_score(val_y, pres))

titan2_mod = RandomForestClassifier(max_features= 'sqrt', max_leaf_nodes= 63, random_state=1)
titan2_mod.fit(imputed_train_X,train_y)

## Next step is to get the titles from each persons name as that is predictive, 
## also adjust hyperperameters in sequnetioal models 
## also nee dto cover the 3rd type where it tells you what was imputed in a later model 

def get_accuracy(max_features, max_leaf_nodes, imputed_train_X, imputed_train_val_X, train_y,val_y, randomstate=1) :
    titan_mod = RandomForestClassifier(max_features=max_features, max_leaf_nodes=max_leaf_nodes, random_state=1)                  
    titan_mod.fit(imputed_train_X,train_y)
    pres = titan_mod.predict(imputed_val_X)
    return(accuracy_score(val_y, pres))
    
for k in [63] :
    print(get_accuracy('sqrt', k, imputed_train_X, imputed_val_X, train_y,val_y))



## submission 



test_path = 'test.csv'
test_data = pd.read_csv(test_path)

incoded = enc.transform(test_data[['Sex','Embarked']])

incod_df = pd.DataFrame(incoded.toarray(), columns= enc.get_feature_names_out())

in_encod_df = incod_df.drop('Embarked_nan', axis=1)

iX_base = test_data.drop(['Sex','Embarked'], axis=1)

inew_full_df = pd.concat([in_encod_df,iX_base],axis =1)

iX = inew_full_df[cols]

iimputed_train_X = pd.DataFrame(my_imputer.transform(iX))
iimputed_train_X.columns = iX.columns

preds = titan2_mod.predict(iimputed_train_X)

submission = pd.DataFrame({'PassengerId' : test_data['PassengerId'], 'Survived' : preds})

submission.to_csv('submit1.csv', index = False)