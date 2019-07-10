import pandas as pd
import numpy as np
import xlrd
import math
import matplotlib.pyplot as plt
#for open excel file
df=xlrd.open_workbook(r"C:\Users\Nitin\Mehasana_EC_f.xlsx")
sheet=df.sheet_by_index(0)
#for see the cell value
sheet.cell_value(0,0)
#to see the number of rows and columns
print(sheet.ncols)
print(sheet.nrows)
#to print the value of any particular value all the rows/column
for i in range (sheet.nrows):
    print(sheet.cell_value(i,0))
#save any particular row value in any dictionary or list
x=[]
for i in range (sheet.nrows):
    x.append(sheet.cell_value(i,0))    
    #make min/max/mean/meddaian/covarince from whole raw or column
min=np.min(x[1:21])
min=np.max(x[1:21])
mean=np.mean(x[1:21])
std=np.std(x[1:21])
meadian=np.median(x[1:21])
cov=np.cov(x[1:21])    
#if u wwant to find variance it list may have to change in matix s
y=np.matrix(x[1:21])
variance =y.var


#for reading csv file
df1=pd.read_csv(r"C:\Users\Nitin\Mehsana_Soil_Health_Carhttps://github.com/jaypadariya/Analytics/blob/master/latitude%20long%20by%20gdal.pyd.csv")
#to read csv data
df1.head()
df1.info()
df1.describe()
df1.columns

#convert column into array
j=df1['EC']
j1=np.array(j)

#reshape columns(it means it can make sure place in proper array foam)
j2=j1.reshape(-1,1)


#plot a graph of two columns 
x=df1[['EC','pH']]
y=np.array(x)
z=y[0:500,0] 
plt.hist(z[0:50],bins=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4])
#bins is the x axes which we can see in the graph values)



#estimation soil property using linear regressionmodel 
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


#for reading csv file
df1=pd.read_csv(r"C:\Users\Nitin\Mehsana_Soil_Health_Card.csv")
#to read csv data
df1.head()
df1.info()
df1.describe()
df1.columns
#corelattion between variables in dataset
df1.corr()


#training a linear regresion model
x1=df1['EC']
y1=df1['pH']
x=np.array(x1[1:20]) #if u want to give limit then it can give in sqaure bracket
y=np.array(y1[1:20])

#fi to the model
x_constant =sm.add_constant(x)
model=sm.OLS(y,x_constant)  #here we have fit model betwwen our y and estimated x

linreg=model.fit()
linreg.summary()

#plot the graph

df1.plot(kind='scatter',x='EC',y='pH')

#plot a regression line
sns.lmplot(x='EC',y='pH',data=df1) #data as it given the file name


#multi linear regression using ststs model

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


#for reading csv file
df1=pd.read_csv(r"C:\Users\Nitin\Mehsana_Soil_Health_Card.csv")
#to read csv data
df1.head()
df1.info()
df1.describe()
df1.columns
#corelattion between variables in dataset
df1.corr()


#training a linear regresion model
x1=df1['EC']
y1=df1[['pH','Fe','Mn']]
x=np.array(x1[1:20]) #if u want to give limit then it can give in sqaure bracket
y=np.array(y1[1:20]) #it may have to pluus 1 coz multiple cant make 

#fi to the model
y_constant =sm.add_constant(y)
model=sm.OLS(x,y_constant)  #here we have fit model betwwen our y and estimated x

linreg=model.fit()
linreg.summary()

#plot the graph

df1.plot(kind='scatter',x='EC',y='Fe')

#plot a regression line
sns.lmplot(x='EC',y='Fe',data=df1) #data as it given the file name



##prediction of with r sq training a multiple regresion model(traininngg skylearn library usin )

import pandas as pd
from sklearn import datasets
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

df1=pd.read_csv(r"C:\Users\Nitin\Mehsana_Soil_Health_Card.csv")

x1=df1['EC']
y1=df1[['pH','Fe','Mn']]
x=np.array(x1[1:20]) #if u want to give limit then it can give in sqaure bracket
y=np.array(y1[1:20]) #it may have to pluus 1 coz multiple cant make 
x=x.reshape(-1,1)
y=y.reshape(-1,3) #here u put 3 bcoz the y have 3 objects pH Fe and Mn
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4) #test size is the size in which do u have to predict if u want to predict 60% u have to write 0.4
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,3)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,3)
lm=LinearRegression().fit(x_train,y_train)
predy=lm.predict(x_test) #prediction
r2=r2_score(y_test,predy) #r square 
print(r2)

model = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])
 # fit to an order-3 polynomial data 
model = model.fit(x, y)
poly_coef = model.named_steps['linear'].coef_ 
print('Polynomial coeeficients : ', poly_coef)




#how to use any paarticular column in for loop

for i in (x1):
       
    
    if i > 0.24:
        print(i,"it is more than o.24")
    elif i < 0.24:
        print("moj")
    else:
        print('jalsa')
        
        

#unsupervised clssification using svm
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
import gdal
import ogr
from sklearn import metrics
from sklearn import svm
import numpy as np
import pandas as pd
from pandas import DataFrame
from timeit import default_timer as timer
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
data=pd.read_csv(r"C:\Users\Nitin\Mehsana_Soil_Health_Card.csv")
x=np.array(data["pH"])
df=DataFrame(data,columns=["Fe","Mn","Zn"])
x=np.array(x[:])
df=np.array(df[:])

x=x.reshape(-1,1)
df=df.reshape(-1,1)
y=df.values
X_train, X_test, Y_train, Y_test = train_test_split(x, y,random_state=0)
X_train=X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,3)
X_test=X_test.reshape(-1,1)
Y_test=Y_test.reshape(-1,3)
svm_model_linear=SVC(kernel="linear",C =1).fit(X_train[:],Y_train[:])
scm_predictions=svm_model_linear.score(X_test[:])
accuracy=svm_model_linear.score(X_test,Y_test)
cm=confusion_matrix(Y_test,svm_predictions)
print("Accuracy of svm is:",accuracy*100)
print("confusion matrix:",cm)
plt.imshow(cm,cmap="hot")

..................................

clf = SVC(gamma='auto')
clf.fit(X_train,Y_train) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
>>> print(clf.predict([[-0.8, -1]]))