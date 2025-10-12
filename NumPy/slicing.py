import numpy as np 

arr = np.array([[10, 11, 12, 13, 14],
               [20, 21, 22, 23, 24],
               [30, 31, 32, 33, 34]])

'''1 Extract the middle row as a flat 1-D array.
(Hint: integer indexing collapses.)

2 Extract the same row but as a 2-D array of shape (1, 5).

3 Extract the middle column as a flat 1-D array.

4 Extract the same column as a true 2-D column vector of shape (3, 1).

5 Slice a 2×3 sub-matrix from the top-left corner.

6 Reverse the order of columns in the whole array (so each row goes [14, 13, 12, 11, 10], etc.).

7 Reverse the order of rows in the whole array.'''

## 1 
print(arr[1, :])

## 2 
print(arr[1:2, :])

## 3 
print(arr[:, 1])

## 4
print(arr[:, 0:1])

## 5 
print(arr[:2, :4])

## 6
print(arr[:,::-1])

## 7 
print(arr[::-1, :])