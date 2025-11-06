import pandas as pd 

temperature = pd.Series([12,18,22,31], index = ['London', 'Atlanta', 'Lisbon', 'Muscat'])

print(temperature)
print(temperature.name,temperature.index, temperature.shape)

"""C2"""

inven1 = pd.Series([30,59,81], index = ['pencils','staples','rubbers'], name = 'Office 1')
inven2 = pd.Series([11,46,97], index = ['pencils','staples','rubbers'], name = 'Office 2')

invendata = pd.DataFrame({'Office 1' : inven1, 'Office 2' : inven2})
print(invendata)
print('---------')
print(invendata.head())
print('---------')
print(invendata.shape)


'''C3'''

Cinv = invendata.to_csv(office_inventory.csv)