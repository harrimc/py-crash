'''Create a 2×5 integer array.

Make three variations:

a slice of the second row

a copy of the first row

Change a few elements in each variation.

After each change, inspect the original array and figure out whether it was affected.

Then confirm your guesses experimentally using any NumPy attributes you think might help.

End by writing one rule in a comment explaining when NumPy reuses memory vs allocates new memory.'''

import numpy as np 

arr = np.array([[1,2,3,4,5],
                [6,7,8,9,10]])
scr = arr[1,:]

cp1 = arr[0,:].copy()

scr[2] = 100  ## expecting 8 to go to 100 

cp1[4] = 81 ## expecting that 5 not go to 81 as its a copy not a view 

print(arr)
print(cp1.base) ##None as its a copy and owns its data
print(scr.base) ## the larger matrix-- not the sub matrix scr


## when its a view whatever you do to the view no matter if its only a part of the larger array, it will still change
## the larger array. But a copy owns its data so its .base is None so any changes to a copy dont affect the original 

reshape = arr[::-1,::-1].copy()
print(reshape)