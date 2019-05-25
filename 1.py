import numpy as np 
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[1,2,3]])
aa = np.dot(a,b.T)
bb = np.dot(b,a.T)
print(aa)
print(bb.T)
