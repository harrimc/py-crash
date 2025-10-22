import pandas as pd

fruits = pd.DataFrame({'Apples' : [30], 'Bannanas' : [21]})
print(fruits)
 

print('--------------------------------------------------------') 
fruit_sales = pd.DataFrame({'Apples' : [35,41], 'Bannanas' : [21,34]}, index=['2017 sales', '2018 sales'])
print(fruit_sales)

print('--------------------------------------------------------')

ingredients = pd.Series([ '4 cups', ' 1 cup', '2 large', ' 1 can'], index = [ 'Flour', 'Milk', 'Eggs', 'Spam'], name = 'Dinner')
print(ingredients)

print('------------------------------------------------------')
 ## reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col = 0 )
''' not running as the csv file is local to Kaggle'''