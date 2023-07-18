# Predicting House Prices: Regression using TensorFlow

💻 Coursera Project Network: Predicting House Prices with Regression using TensorFlow

## Evaluating House Prices Based on the Following Features:

### Input Features

1. Year of sale of the house
2. Age of the house at the time of sale
3. Distance from the city center
4. Number of stores in the locality
5. Latitude
6. Longitude

## Steps:

1. Importing Libraries:
   - TensorFlow
   - Pandas
   - Matplotlib.pyplot
   - Utils
   - Sklearn.model_selection: train_test_split
   - Tensorflow.keras.models: Sequential
   - Tensorflow.keras.layers: Dense, Dropout
   - Tensorflow.keras.callbacks: EarlyStopping, LambdaCallback

2. Importing Data:
   - Data imported from a CSV file using the Pandas library.

3. Data Normalization:
   - Apply Z-score normalization to obtain normalized values and bring features into the same range.

4. Creating Training and Testing Sets:
   - Splitting the given data into training and testing sets using the Tensorflow library's split function.

5. Create Model:
   - Building a model using the Sequential library from Keras, consisting of 3 layers and 1 hidden layer.

6. Model Training:
   - Training the model using the training data, utilizing a loss function and optimizer.

7. Predictions:
   - Validating the predictions and evaluating accuracy with the results.

## License:
All rights belong to Coursera.
