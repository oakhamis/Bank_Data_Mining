# Bank Term Deposit Prediction

This project aims to predict bank term deposits using various machine learning algorithms.

## Prerequisites

The following R libraries are required:

- class
- e1071
- caret
- rpart.plot
- Boruta
- ggplot2
- ranger
- dplyr
- corrplot
- pROC
- reshape2
- shiny
- xgboost

## Dataset Description

- **File**: `bank-full.csv`
- **Description**: Main dataset used for training and evaluation.
  
- **File**: `DatasetTable.csv`
- **Description**: Provides a detailed description of the attributes present in the main dataset.

## Data Pre-processing

1. Check for missing values in the dataset.
2. Identify and handle duplicated rows.
3. Convert the target variable 'y' to a binary format (0 for "no" and 1 for "yes").

## Exploratory Data Analysis (EDA)

1. Plot histograms for numeric attributes.
2. Visualize the distribution of the target variable 'y'.
3. Generate a correlation matrix for numeric variables.
4. Display bar plots for categorical variables.

## Model Building and Evaluation

The following models are trained and evaluated:

### Logistic Regression
- A generalized linear model (GLM) with a binomial family.

### Decision Tree
- A recursive partitioning method using the rpart library.

### Random Forest
- An ensemble learning method that constructs a multitude of decision trees at training time.

The results of the models are then compared in terms of accuracy, sensitivity, specificity, and balanced accuracy.

## Visualization

1. Feature importance from the Random Forest model.
2. Comparison of model performance metrics using bar plots.

