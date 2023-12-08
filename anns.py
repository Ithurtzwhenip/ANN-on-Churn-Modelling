import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Using ColumnTransformer for one-hot encoding
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), [1])  # Specify the index of the categorical column
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)

# Avoiding the dummy variable trap
X = X[:, 1:]
print(X.shape[1])
# Now X should contain the encoded categorical features
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)  # Set with_mean to False
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Create your classifier here
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dropout(rate=0.1))
# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

# Example:
# Use our ANN model to predict if the customer with the following informations will leave the bank:
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: $ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: $ 50000


new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)

if new_prediction:
    print("The customer is likely to leave the bank.")
else:
    print("The customer is likely to stay with the bank.")

# pip install scikeras

# Evaluating the ANN
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
  classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
  classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
  classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
  return classifier

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10, epochs=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train, cv=10)

mean= accuracies.mean()
variance=accuracies.std()
print(mean,variance)

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Define the function with all the parameters from the param_grid
def build_classifier(optimizer='adam', batch_size=10, epochs=100):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Create the KerasClassifier with the default parameters
classifier = KerasClassifier(build_fn=build_classifier)

# Specify the parameters in the param_grid
parameters = {'optimizer': ['adam', 'rmsprop'],
              'batch_size': [25, 32],
              'epochs': [100, 500]}

# Use GridSearchCV with the corrected parameters
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)

# Fit the grid search to the training data
grid_search = grid_search.fit(X_train, y_train)

# Get the best parameters and best accuracy
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Parameters:", best_parameters)
print("Best Accuracy:", best_accuracy)

