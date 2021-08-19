import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("/Users/nikhilgautam/Desktop/33/Real estate.csv",sep=',')

data = data[['X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','Y house price of unit area']]

predict = "Y house price of unit area"
x = np.array(data.drop(predict,axis=1))
y = np.array(data[predict])
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)
sc =model.score(X_test,Y_test)
print(sc)
predictions = model.predict(X_test)

df = pd.DataFrame({'Actual':Y_test,'Predicted':predictions})
print(df.head())
#end