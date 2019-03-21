#importing libraries for data manipulation and analysis
import pandas as pd 
import numpy as np                     
import matplotlib.pyplot as plt 
import seaborn as sns 

#Reading the dataset
df = pd.read_csv('data.csv')

#considering features on x axis and price on y axis
x= df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']]

y = df['price']       


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 101 ) 

from sklearn.linear_model import LinearRegression

LM = LinearRegression()

LM.fit(x_train, y_train)

predictions = LM.predict(x_test)

plt.scatter(y_test, predictions)

plt.show()
