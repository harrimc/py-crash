import pandas as pd   ### notes of continuance at the bottom 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

train_path = 'train.csv'
traindf = pd.read_csv(train_path)


## make the column transformer and preprocessor

## the list stuff is kinda uneeded as I just put it in manually but its a good example of what to do with larger Df'
catlist = (traindf.dtypes == 'object')
indcatlist = list(catlist[catlist].index)
ref_indcatlist = ['Sex', 'Embarked']

numlist = ((traindf.dtypes == 'int64') | (traindf.dtypes =='float64'))
indnumlist = (numlist[numlist].index)
ref_indnumlist = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

y = traindf.Survived
X = traindf[['Sex', 'Embarked','Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]     # double [[ ]] as otherwie it thinks its like a tuple key 


numerical_transformer = SimpleImputer(strategy='mean')

catagorical_transformer = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')), 
                                          ('OH_encoding', OneHotEncoder(handle_unknown= 'ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, ref_indnumlist),
                                               ('cat', catagorical_transformer, ref_indcatlist)])

## now split data

## now make the secondary scope pipeline

model = RandomForestClassifier(n_estimators=100,max_features= 'sqrt',random_state=1)

my_model = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])

## now we use the cv before we fit the data 

cV = StratifiedKFold(n_splits = 5, shuffle= True, random_state = 1)

scores = cross_val_score(my_model, X, y, cv=cV, scoring='accuracy')

print(scores, scores.mean())

my_model.fit(X,y)

## submit


test_path = 'test.csv'
testdf = pd.read_csv(test_path)
test_X_val = testdf[['Sex', 'Embarked','Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

test_preds = my_model.predict(test_X_val)

submission = pd.DataFrame({'PassengerId' : testdf['PassengerId'], 'Survived' : test_preds})

submission.to_csv('submit3.csv', index= False) 