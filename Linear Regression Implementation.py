import numpy as np
import matplotlib.pyplot as plt

x =2*np.random.rand(100,1)
y=4+3*x+np.random.rand(100,1)
plt.plot(x,y,"b")
plt.plot(x,y,".b")    # .b for points

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)   # fit is the training function to perform training
print(model.coef_)
print(model.intercept_)