import numpy as np 

'''1. Create a base array of shape (2, 5).

2. Slice the second row into a variable v1.

3. Print whether it’s a view or copy using .base.

4. Modify one element of v1 and show the change in the original array.

5. Now make an explicit copy c1 = v1.copy().

6. Print whether it’s a view or copy using .base.

7. Modify c1 and show that the original array stays unchanged.

8. Finally, create a second view by slicing a column; modify it and check again'''


## 1 
arr = np.array([[1,2,3,4,5],
                [6,7,8,9,10]])

## 2 
v1 = arr[1,:]

## 3
print(v1.base)

## 4
v1[2] = 50 
print(arr) ##expect that 8 will go to 50

## 5 
c1 = v1.copy() 

## 6 
print(c1.base)

## 7 
c1[1] = 20 
print(v1)

## 8 
col = arr[:,2:3 ]
col[1,0] = 90 
print(col) 
print(arr)  ## expect that 8 will go to 90

