import keras
import numpy

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
X,Y=make_regression(n_samples=100,n_features=2,noise=0.1,random_state=1) #for taking random value
ScalerX,ScalerY=MinMaxScaler(),MinMaxScaler()  #scaling the x and y coz the values we have taken is very high
ScalerX.fit(X)
ScalerY.fit(Y.reshape(100,1))  #it have to reshape coz complete 2d matrix is not y..it may have to make([100,1)
X=ScalerX.transform(X)   #define to one variable that can see in variable exploral
Y=ScalerY.transform(Y.reshape(100,1))  #it is define to see the reshaped y

model = Sequential()  #cal the sequential model
model.add(Dense(4, input_dim=2, kernel_initializer='normal', activation='relu'))  #(12 is the neuron,input dimension is 2-D )
model.add(Dense(4, activation='relu'))  #again we take 4 neaurons and relu activation function
model.add(Dense(1, activation='linear'))  #output layer which has alwys linear function and nural is alwys 1
model.compile(loss='mse',optimizer='adam') #cross antropy is used in classification as a optimizer
model.fit(X,Y,epochs=5,verbose=0)  #fit the model from X and Y

Xnew,a=make_regression(n_samples=3,n_features=2,noise=0.1,random_state=1) #again define values randomly
Xnew=ScalerX.transform(Xnew) #xnew shows negative vlaurs so it may have to scale in 0 to 1
Ynew=model.predict(Xnew) #define xnew in ynew for show
a=ScalerY.transform(a.reshape(3,1)) #reshape and transform a with yscaler it make 2d array
#mean scale error 
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(a, Ynew) # we can make mean square error of a and Ynew
