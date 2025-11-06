import pandas as pd

df = pd.DataFrame({'country' : ['Italy','Portugal','US','US'], 'Province' : ['Sicily & Sardinia','Douro','California','New York'], 'region_1' : ['Etna', None,'Napa Valley','Finger Lakes'], 'region_2' : [None, None,'Napa','Finger Lakes']}, index =[0,1,10,100])

print(df)
     
### 9 Problems solved on Kaggle, hard to do dataframes on here if I dont have the CSV file 


reviews = pd.DataFrame({'country' : 'Australia', 'score' : 98})
top_oceania_wines = reviews.loc[((reviews.country == 'Australia') | (reviews.country == 'New Zealand')) & reviews.score >= 95]

