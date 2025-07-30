# California-Median-Housing-Price-Prediction
## Overview
This project aims to study the california housing price data (available in the repository) in 6 sections:
### Section 1: Import required libraries
Essential Python libraries are imported to support data manipulation, visualization, modeling, and evaluation.
### Section 2: Descriptive analysis of the dataset 
This section explores the dataset through:
* Histograms to understand feature distributions
* Correlation analysis to assess relationships between variables

### Section 3: Build and assess multiple linear regression models
#### 3.1 A simple linear model using all relevant predictors 
#### 3.2 Potential issues in the initial fitted model
#### 3.3 Improvements on the initial MLR model
Improvements include:
* Feature selection
* Adding interaction term
* Ridge regression
* Principal Component Regression
#### 3.4 Comparing models using training and estimated test MSE
#### 3.5 Assessing model performance with and without censoring the perdicted values
#### 3.6 Compute test MSE using the selected best MLR model

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
In summary, among all the models explored, the K-Nearest Neighbors (KNN) regression model achieved the lowest test mean squared error, indicating the best predictive performance on the test set. (Note: while KNN performed best in this case, its sensitivity to local data structure may limit generalizability in other contexts.)<br>
Limitations of this analysis include: 
* Rows with missing values were removed, which may introduce bias.
* A limited number of model types were explored, which may restrict the comprehensiveness of the performance comparison.
