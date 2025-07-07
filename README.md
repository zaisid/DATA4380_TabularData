![](UTA-DataScience-Logo.png)

# Parkinson's Disease Prediction

* This repository holds an attempt to apply machine learning techniques to model and predict the presence of Parkinson's Disease using data from the [UCI Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons). 

## Overview

* This section could contain a short paragraph which include the following:
  * **Definition of the tasks / challenge**  Ex: The task, as defined by the Kaggle challenge is to use a time series of 12 features, sampled daily for 1 month, to predict the next day's price of a stock.
  * **Your approach** Ex: The approach in this repository formulates the problem as regression task, using deep recurrent neural networks as the model with the full time series of features as input. We compared the performance of 3 different network architectures.
  * **Summary of the performance achieved** Ex: Our best model was able to predict the next day stock price within 23%, 90% of the time. At the time of writing, the best performance on Kaggle of this metric is 18%.

## Summary of Workdone

### Data

* Data:
  * Type: Tabular data from analysis of various voice recordings. 
  * Size: 195 rows with data from 32 individuals, 22 numerical features, 1 binary target variable 
  * Instances (Train, Test, Validation Split): how many data points? Ex: 1000 patients for training, 200 for testing, none for validation

#### Preprocessing / Clean up

Initial preprocessing included removal of the ID column, class balancing through oversampling rows, and scaling data. Any outliers observed were kept in the data since 

#### Data Visualization





### Problem Formulation

* Define:
  * The inputs were the various voice and audio measures from voice recordings taken in the original patient study, which include frequency, amplitude, and pitch. These were used to determine whether an individual had Parkinson's or not.
  * Models used:
    * Decision Tree
    * K-Nearest Neighbor (KNN)
    * Random Forest (for aggregated data)
  * Loss, Optimizer, other Hyperparameters.

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

In the future, I'd like to apply 

## How to reproduce results

To reproduce the analysis and modeling results:

* Clone this repository.
* Open Tabular_Prototype_ZS.ipynb using Jupyter Notebook or Jupyter Lab.
* Download dataset and the tabular_preproccessing.py module in the same directory as the notebook.
* Run all cells.
* Repeat with Tabular_Aggregated.ipynb
All necessary data loading, preprocessing, and model evaluation steps are included in the notebooks and module.


### Contents of Repository

* Tabular_Feasibility_ZS.ipynb: Initial exploratory data analysis of data and some preprocessing, including a preliminary baseline model
* tabular_preprocessing.py: module containing all preprocessing steps done in Tabular_Feasibility_ZS.ipynb
* Tabular_Prototype_ZS.ipynb: contains the first round of modelling
* Tabular_Aggregated.ipynb: 

### Software Setup
All modelling and data manipulation was done using scikit-learn, pandas, and numpy. Majority of visualizations were completed with matplotlib.

### Data

The data can be downloaded on its [UCI webpage](https://archive.ics.uci.edu/dataset/174/parkinsons).

## Citations

Little, Max. "Parkinsons." UCI Machine Learning Repository, 2007, https://doi.org/10.24432/C59C74.





