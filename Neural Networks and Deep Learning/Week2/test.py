import numpy as np
a=np.random.randn(200,400)
b=np.random.randn(400,3)
c=np.sum(a,axis=0,keepdims=True)

print(c.shape)
# d=a*b
# print(d.shape)