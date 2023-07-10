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
x=df_norm.iloc[:, :6]
x.head() #prints out the table withoud the price 

#select labels for the data table:
y= df_norm.iloc[:,-1]
y.head() #seperated prices table 

#features and label values:
x_arr =x.values
y_arr = y.values
print("features array shape:" + x_arr.shape)
print("labels array shape:" + y_arr.shape)

#splitting the data: Trainig + Testing set:
x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.05, random_state=0) #test size : 5% of the total data
print("training set: ", x_tain.shape, y_train.shape) #training set shape 
print("training set: ", x_tain.shape, y_train.shape) #testing set shape

#MODEL TRAINING:
#Create a model:
# using 3 neural network architecture - relu activaation function (on all layers - expect the output layer) 
def get_model():
    #using sequential class from keras- can pass in list of layers to create an architecture
    #3hiddden layers
    #laye 1 - Dense (10 nodes input - list of 6 value, activation fxn : rectified linear unit
    #layer 2 &3 - Dense with 20 and 5 nodes , activation function = relu
    #layer 4 - output 

    # all the layers are fully connected layers - no of parameters= no.of nodes
   #creating the model
    model = Sequential([
    Dense(10, input_shape = (6,), activation ='relu'), 
    Dense(20, activation ='relu'),
    Dense(5, activation = 'relu'),
        Dense(1) #as its a regression problem just need linear values - no actv. fxn requried
    ])

#compiling the model
model.complie(
    #Optimiser: minimize loss function:
    loss = 'mse', #loss function mean square error
    optimizer = 'adam' #optimization: adam function
)
    return model

#summarizing the model:
get_model().summary()
#calculaiton of paradms

#MODEL TRAINING:
#earlyStopping - call back from keras , wait for (patience) seconds if val loss not changing then stops the training:
es_cb = EarlyStopping(monitor='val_loss', patience =5)
#create  a model using get model function:
model = get_model()
preds_on_untrained=model.predict(x_test)



