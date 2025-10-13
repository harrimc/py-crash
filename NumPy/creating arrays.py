import numpy as np 

oned_array = np.array([1, 2, 3])
twod_array = np.array([[1, 2, 3], [4, 5, 6]])
threed_array = np.array([[[1,2,3],[4,5,6]], [[4, 7, 6], [9,8,1]]])
sixd_array = np.array([1,2,3], ndmin = 6)

print(oned_array.ndim)
print(twod_array.ndim)
print(threed_array.ndim)
print(sixd_array.ndim)


