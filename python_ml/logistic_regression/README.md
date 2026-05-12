# 1. Logistic Regression From Scratch

## Project Overview

This project demonstrates how to implement a logistic regression model from scratch using Python and NumPy.

The main goal is to understand how logistic regression works internally without using machine learning libraries such as `scikit-learn`.

In this project, logistic regression is implemented manually using:

- sigmoid function;
- cost function;
- gradient calculation;
- gradient descent;
- decision boundary visualization.

The model predicts whether a student is admitted based on two exam scores:

- Exam 1 score;
- Exam 2 score.

The output is a binary classification result:

0 = Not admitted
1 = Admitted

_________________________________________

# 2. Regularized Logistic Regression With Polynomial Features

# Regularized Logistic Regression With Polynomial Features

## Project Overview

This project demonstrates how to implement regularized logistic regression with polynomial features from scratch using Python and NumPy.

The main goal is to understand how feature mapping and regularization affect logistic regression when the data cannot be separated well with a straight line.

In the previous logistic regression project, the model used only two original input features and created a linear decision boundary.

In this project, the model uses:

- sigmoid function;
- cost function;
- gradient calculation;
- gradient descent;
- polynomial feature mapping;
- lambda regularization;
- nonlinear decision boundary visualization.

The model predicts whether a microchip is accepted or rejected based on two test results:

- Microchip Test 1;
- Microchip Test 2.

The output is a binary classification result:

text
0 = Rejected
1 = Accepted

__________________________________

# 3. Logistic Regression Using Scikit-Learn

## Project Overview

This project demonstrates how to train a logistic regression model using `scikit-learn`.

The main goal is to compare a manual implementation of logistic regression with a library-based implementation.

In the previous projects, logistic regression was implemented from scratch using:

- sigmoid function;
- cost function;
- gradient calculation;
- gradient descent;
- regularization;
- decision boundary visualization.

In this project, the same type of model is trained using the built-in `LogisticRegression` class from `scikit-learn`.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-Learn