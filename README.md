# California-Median-Housing-Price-Prediction

## Table of Contents
- [Python packages requirements](#python-package-requirements)
- [Overview](#overview)
- [Summary and limitations](#summary-and-limitations)

## Python package requirements
The project requires packages which are sepcified in the requirement.txt and can be installed through:
```
pip install -r requirements.txt
```

## Overview
This project aims to study the california housing price data (available in the repository) in 6 sections:
### Section 1: Import required libraries
Essential Python libraries are imported to support data manipulation, visualization, modeling, and evaluation.
### Section 2: Preliminary analysis of the dataset 
This section explores the dataset through:
* Histograms to understand feature distributions
* Correlation analysis to assess relationships between variables

### Section 3: Build and assess multiple linear regression models
#### 3.1 Building MLR models
Choices include:
* Full model using all relevant prodictors
* Feature selection
* Adding interaction term
* Ridge regression
* Principal Component Regression
#### 3.2 Compare best two models using training and estimated test MSE
Best two models selected through training mean squared error (MSE). Estimated test MSE calculated through:
* 80-20 split validation set approach
* 5-fold cross validation
* Leave-One-Out cross validation

Assess whether additional censoring applied to the predicted values improves model performance.
#### 3.3 Compute test MSE using the selected best MLR model

### Section 4: Compare Generalized Additive (GAM) Model with K-Nearest Neighbors (KNN) regression model

### Section 5: Build and assess classification models
Models selected:
* Logistic regression model
* KNN classificaiton model

Model performance is evaluated using ROC curves and classification error rates.

### Section 6: Combination of regression model and classifer
Explore a two-stage pipeline:
* Classify observations into relevant groups
* Apply regression to a subset (e.g., class 1)

Assess the effectiveness of this approach using test mean squared error (MSE).

## Summary and limitations
In summary, among all the models explored, the K-Nearest Neighbors (KNN) regression model achieved the lowest test MSE, indicating the best predictive performance on the test set. (Note: while KNN performed best in this case, its sensitivity to local data structure may limit generalizability in other contexts.)

Limitations of this analysis include: 
* Rows with missing values were removed, which may introduce bias.
* A limited number of model types were explored, which may restrict the comprehensiveness of the performance comparison.
