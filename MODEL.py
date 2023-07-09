#1. IMPORTING LIBRARIES:
import pandas as pd #python tool- for data analysis 
import matplotlib.pyplot as plt
import tensorflow as tf

from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback

%matplotlib inline
tf.logging.set_verbosity(tf.logging.ERROR)

print('Libraries imported.')

#2.IMPORTING THE DATA:
df = pd.read_csv('data.csv', names=column_names) #padas return ds-dataframes +set column names\
df.head() #returns 5 rows from dataset
#Checking for missing data:
df.isna.sum() #isan return either true or false for each column + sum (as the number of columns are more we prefer taking the total as a whole)

#3.NORMALISATION
df = df.iloc[:, 1:] #IGNORING THE FIRST ROW all rows selected and first column ex. #rows, columns
df_norm = (df-df.mean())/df.std() #normalising 
df_norm.head()
#labels:
#predicted value to back normal form 
y_mean = df['price'].mean() #mean(orginal distribution)
y_std = df['price'].std()

#defining a function:
def convert_label_value(pred):
    return int(pred*y_std+y_mean) #y=mx+c format

print(convert_label_value(0.350088)) #checking the pred.

#4. TRAINING AND TEST SETS:
#REMOVE price column - as(price is the ouput that we want to predict) 
x=df.norm.iloc[:, :6]
x.head() #prints out the table withoud the price 

#select labels for the data table:
y= df_norm.iloc[:,-1]
y.head() #seperated prices table 

#features and label values:
x_arr =x.values
y_arr = y.values
print("features array shape:" + x_arr.shape)
print("labels array shape:" + x_arr.shape)



