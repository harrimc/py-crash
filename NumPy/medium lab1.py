'''Create a 2-D NumPy array of integers 0–23 shaped 4×6.
Print its shape, size, and number of dimensions.

Print the second row and the last two columns (as separate outputs).
Create a subarray containing rows 1 and 3, columns 2 through 5 (exclusive of 5).
Print its shape, then change one element inside this subarray.
Print the original 4×6 array to verify whether it changed.

Create a copy of that subarray and change a different element inside the copy.
Print the original 4×6 array again to verify whether it changed this time.

Convert the copied subarray to float64 with .astype() and print its dtype.
Flatten this float array back to 1-D using reshape(-1) and print the first 4 elements.

Reshape the original 0–23 data into 2×3×4 and print its shape and ndim.
Print one deep element from this 3-D array (any valid coordinate).

Iterate over the 3-D array with  and print only the first five scalars encountered.
Repeat iteration with flags=['buffered'] and print the first five elements (as Unicode).

Print one summary line showing the shape transitions you performed in order.
'''
import numpy as np 

kar = np.arange(24)
narr = kar.reshape(4,6)
print(narr)
print(narr.shape,narr.size,narr.ndim)


print(narr[1,-2:])
slic = narr[0::2,1:4]
print(slic.shape)
slic[1,1] = 110
print(narr)     ## the 14 in narr should go to 110. as slic is a a 2x3 [[1,2,3],[14,14,15]]. [1,1] is 14 

print('-------------------------------------------')
copslic = slic.copy()
copslic[0,2] =96 
print(narr)    ## wont change as its a copy but if it wasnt then 3 would go to 96
coppsl = copslic.astype('f8')
print(coppsl.dtype)
print(coppsl.reshape(-1)[0:3])   

rarr = kar.reshape(2,3,-1)
print(rarr.shape,rarr.ndim)
print(rarr[0,2,1])            ## its a 2x3 matrix so this would be the third element top row, the value should be 9 as 0 if the first number
for k in np.nditer(rarr) :
    print(k)

for k in np.nditer(rarr,flags = ['buffered'], op_dtypes=('U')) :
    print(k)
