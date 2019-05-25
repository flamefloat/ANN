import numpy as np 
a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,0])



print(b.shape)
b = b.reshape(1,2)
print(b.shape)
c = b.T
print(np.exp(-c))

