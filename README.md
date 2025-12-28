# Artificial Neural Network for Electrical Energy Output Prediction

This project demonstrates the use of an **Artificial Neural Network (ANN)** to predict electrical energy output based on input features. The notebook walks through the full pipeline of data preprocessing, ANN construction, training, and prediction evaluation.

---

## Table of Contents

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Building the ANN](#building-the-ann)  
5. [Training the ANN](#training-the-ann)  
6. [Conclusion](#conclusion)  

---

## Overview

The goal of this project is to predict electrical energy output using an ANN implemented with **TensorFlow / Keras**. The model consists of:

- Input layer  
- Two hidden layers with 6 units each and ReLU activation  
- Output layer for regression prediction (no activation function)

The ANN is trained using the **Adam optimizer** with mean squared error as the loss function.

---

## Dataset

The dataset (`electrical_energy_output.xlsx`) contains multiple features (input variables) and a single target (electrical energy output).  
- `X`: Input features  
- `y`: Target output  

The dataset is split into **training** and **test** sets using an 80/20 split.

---

## Data Preprocessing

- Import the dataset with `pandas`  
- Split data into features (`X`) and target (`y`)  
- Perform train-test split using `sklearn.model_selection.train_test_split`

---

## Building the ANN

1. Initialize the ANN using `Sequential()`  
2. Add an input layer and first hidden layer with ReLU activation  
3. Add a second hidden layer with ReLU activation  
4. Add an output layer with one unit (for regression)

> Note: For classification tasks, use `sigmoid` (binary) or `softmax` (multi-class) activation in the output layer.

---

## Training the ANN

- Compile the ANN with:
  - **Optimizer:** Adam  
  - **Loss function:** Mean Squared Error (MSE)  

- Train the ANN on the training set:
  - **Batch size:** 32  
  - **Epochs:** 100  

---

## Conclusion

The Artificial Neural Network demonstrates strong predictive performance on the test set, achieving an R² score of 0.92, which indicates that the model explains the majority of the variance in electrical energy output. The Mean Squared Error (24.70) and Mean Absolute Error (3.96) further confirm that prediction errors remain limited and acceptable for a regression task of this nature.

Residual analysis shows errors centered around zero with no visible systematic pattern, suggesting that the model does not suffer from bias and that its predictions are well-balanced across the range of outputs. This behavior indicates good generalization and confirms that the model has successfully captured the underlying relationships in the data.

Overall, the results validate the effectiveness of the ANN architecture for this problem, providing both high accuracy and stable error behavior, making it suitable for practical energy output prediction tasks.
