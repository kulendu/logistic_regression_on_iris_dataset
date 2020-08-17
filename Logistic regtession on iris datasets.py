'''
    This regression model is made only for practice purpose 
    where the user can give certain inputs, and the model will
    predict the O/P as 3 types of plant.

'''

#importing the necessary dependencies
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris  #importing the iris datasets

iris_data = load_iris() #loading the iris dataset
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
# df.head()


#Making a dataframe with the target datas
df_target = pd.DataFrame(data=iris_data.target, columns=["species"])    

#concating the df and the df_target in new_df
new_df = pd.concat([df,df_target], axis=1)
new_df.shape 

#importing the train_test_split module
from sklearn.model_selection import train_test_split


x = new_df.drop('species', axis=1)
y = new_df['species']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)

#importing the Logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)   #fitting the data in the model

pred = log_reg.predict(x_test)  #predicting the output



from sklearn.metrics import classification_report
print(classification_report(y_test, pred))  #Analysing the classification report
print(log_reg.predict([[5.1,3.5,5.4,1.2]]))






