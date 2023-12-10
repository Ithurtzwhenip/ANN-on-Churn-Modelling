# ANN-on-Churn-Modelling

# Deep Learning with Artificial Neural Networks (ANN) for Customer Churn Prediction

This repository contains code for creating and tuning an Artificial Neural Network (ANN) model using TensorFlow and Keras to predict customer churn. The dataset used for this project is the "Churn_Modelling.csv" dataset, and the goal is to determine whether a customer is likely to leave the bank based on various features.

## Getting Started

### Prerequisites

Make sure you have the required libraries installed. You can install them using the following:

```bash
pip install pandas tensorflow keras matplotlib scikeras
```

### Dataset

The dataset used in this project is named "Churn_Modelling.csv." It contains information about customers, including features such as credit score, geography, gender, age, tenure, balance, and more.

### Running the Code

1. Clone the repository:

   ```bash
   git clone https://github.com/Ithurtzwhenip/ANN-on-Churn-Modelling
   cd ANN-on-Churn-Modelling
   ```

2. Run the Jupyter Notebook or Python script to see the model in action:

   ```bash
   jupyter notebook ann.ipynb
   ```

   or

   ```bash
   python ann.py
   ```

## Code Overview

### Data Preprocessing

- Load the dataset using Pandas.
- Encode categorical features using Label Encoding and One-Hot Encoding.
- Split the dataset into training and testing sets.
- Scale the features using StandardScaler.

### Building the ANN

- Create a Sequential model using Keras.
- Add input and hidden layers with dropout to prevent overfitting.
- Compile the model using the Adam optimizer and binary crossentropy loss for binary classification.

### Training the ANN

- Fit the model to the training data.

### Making Predictions

- Use the trained model to predict customer churn on the test set.
- Make predictions for a new customer based on provided information.

### Evaluating the Model

- Use cross-validation to evaluate the model's performance.
- Tune hyperparameters using GridSearchCV to find the best combination.

## Prediction for a New Customer

To use the trained model for predicting whether a new customer will leave the bank, provide the customer's information:

```python
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

if new_prediction:
    print("The customer is likely to leave the bank.")
else:
    print("The customer is likely to stay with the bank.")
```

## Model Evaluation

The accuracy and variance of the model are evaluated using cross-validation:

```python
mean_accuracy = accuracies.mean()
variance_accuracy = accuracies.std()
print("Mean Accuracy:", mean_accuracy)
print("Variance:", variance_accuracy)
```

## Hyperparameter Tuning

GridSearchCV is used to find the best hyperparameters for the model:

```python
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best Parameters:", best_parameters)
print("Best Accuracy:", best_accuracy)
```

Feel free to explore and modify the code based on your requirements.
