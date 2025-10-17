import numpy as np 

'''Create a 2-D NumPy array with integers 0–11, shaped 3×4.
Print its shape, size, and number of dimensions.

Iterate once over the array normally, printing each row.

Then iterate again with nested loops to print every element individually.

Now use np.nditer() to loop through all elements as scalars.

Repeat using flags=['buffered'] and op_dtypes=['U'] to iterate over the same array as Unicode.

Finally, reshape the array into 2×3×2 and iterate once more, printing only the first five elements.

Print one summary line showing how many iterations occurred in each method.
'''

arr = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12]])
print(arr.shape)
print(arr.size)
print(arr.ndim)

for x in arr :
    print(x)

for y in arr :
    for x in y:
        print(x)
    
for k in np.nditer(arr):
    print(k)

for k in np.nditer(arr,flags = ['buffered'], op_dtypes=['U']) :
    print(k)


newarr = arr.reshape(2,3,-1).copy()
print(newarr.shape)

