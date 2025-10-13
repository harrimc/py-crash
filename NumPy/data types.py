
import numpy as np 

''' Create one array of integers and one of floats.

Check their dtype and itemsize.

Convert the float array to int and print both arrays.

Make a string (U) array from [1, 20, 300] and check its dtype.

Convert that string array back to integers.

Create an array [0, 1, -5, 3], convert it to bool, and print results.

Print a one-line summary of all dtypes used.'''

arr = np.array([[1,2,3,4,5],
                [6,7,8,9,0]])

farr = np.array([[1.1,1.2,1.3,1.4,1.5],
                 [1.6,1.7,1.8,1.9,2.0]])

## 1 
print(arr.dtype)
print(arr.itemsize)

print(farr.dtype)
print(farr.itemsize)

## 2 

newfarr = farr.astype('i4')
print(newfarr.dtype)

## 3
yolo = np.array([1,20,300], dtype = 'U')

print(yolo.dtype)

## 4 

yolo = yolo.astype('i4')
print(yolo.dtype)

## 5 
bow = np.array([0, 1, -5, 3])

newbow = bow.astype('b1')
print(newbow.dtype)
print(newbow)

## 6 
print('dtype can be used in 2 ways: 1, to check what data type a array is.' \
' 2, to set the data type of a rray when it is being defined.'
' .astype can be used to create a copy of an array using a different variable which has a differnt data type')