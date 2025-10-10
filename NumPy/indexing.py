import numpy as np 

arr = np.array([[10, 11, 12, 13],
                [20, 21, 22, 23],
                [30, 31, 32, 33]])

'''Print element in row 1, col 2.

Print last element using negative indexing.

Print entire second row.

Print entire third column. '''

print(arr[0,1])
print(arr[-1,-1])
print(arr[1, 0:4])
print(arr[0:, 2])