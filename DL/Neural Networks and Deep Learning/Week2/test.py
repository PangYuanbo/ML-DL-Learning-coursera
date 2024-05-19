import numpy as np
a=np.random.randn(200,400)
b=np.random.randn(400,3)
c=np.sum(a,axis=0,keepdims=True)
b2=np.zeros((1,3))
print(c.shape)
print(b2.shape)
# d=a*b
# print(d.shape)