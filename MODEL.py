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
