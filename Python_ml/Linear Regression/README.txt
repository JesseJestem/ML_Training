# Simple Rain Probability Prediction Based on Humidity

## Project Overview

This project is a simple machine learning exercise that predicts the probability of rain based on humidity level.

The main goal of this project is not to build a production-ready weather prediction model, but to practice the core ideas of machine learning:

- preparing training data;
- normalizing input values;
- creating polynomial features;
- implementing the cost function manually;
- implementing gradient descent manually;
- visualizing the trained model with Matplotlib.

I used humidity as the input feature because it is intuitive, easy to understand, and makes it easier to track possible mistakes in the code.

---

## What the Model Does

The model takes humidity values as input and predicts a rain probability value between `0` and `1`.

Example interpretation:

Humidity: 20%  -> low rain probability
Humidity: 75%  -> higher rain probability
Humidity: 95%  -> very high rain probability