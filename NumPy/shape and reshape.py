'''Create a 1-D array of integers 0–23 using NumPy.

Reshape it into a 3×8 array and print its shape, number of dimensions, and total size.

Print the element at row 2, column 5.

Reshape the same data into 4×6 and print its shape.

Reshape the data again into 2×3×4 and print its shape.

Print one element deep inside this 3-D array (any coordinate you choose).

Finally, reshape everything back to 1-D and print the shape once more.

Print one clean summary line showing each shape transition in order. '''

import numpy as np

arr = np.arange(24)

Br = arr.reshape(3,8)
print(Br.shape)
print(Br.size)
print(Br.ndim)

print(Br[1,4])

Ne = Br.reshape(4,6).copy()
print(Ne.shape)

Am = Ne.reshape(2,3,4).copy()
print(Am)
print(Am[1,2,1])

flat = Am.reshape(-1)
print(flat)

## summery 
## (24,) -- (3,8) -- (4,6) -- (2,3,4) -- (24,)
##  1d        2d.      2d.        3d.      1d 