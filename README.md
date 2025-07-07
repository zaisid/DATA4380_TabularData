![](UTA-DataScience-Logo.png)

# Parkinson's Disease Prediction

* This repository holds an attempt to apply machine learning techniques to model and predict the presence of Parkinson's Disease using data from the [UCI Parkinson's dataset](https://archive.ics.uci.edu/dataset/174/parkinsons). 

## Overview

This project tackles the task of detecting Parkinson’s disease using a small tabular dataset derived from vocal measurements of 32 individuals. The original dataset includes 22 numerical features per sample, with each sample tied to a voice recording from one of the individuals. The binary classification task is to determine whether the speaker has Parkinson’s based on their vocal features.
The project was carried out in two phases:

Initial Modeling: Models were trained on the original dataset using a standard train/test split and oversampling to address class imbalance. Despite achieving high precision and ROC-AUC scores, these results were likely inflated due to multicollinearity, data leakage, and overfitting.

Aggregated Modeling: To improve generalizability, data was aggregated by patient using mean and variance of each feature. Models were trained using a Leave One Out approach to reflect the small sample size of the data. Random Forest performed best in this phase, with an accuracy of 0.844, though results may still be influenced by class imbalance.

Across both phases, feature selection, scaling, and multiple model types (Decision Tree, KNN, Random Forest) were used and compared. Ultimately, while early models appeared promising, the aggregated approach yielded more realistic and generalizable performance.






## Summary of Workdone

### Data

* Type: Tabular data from analysis of various voice recordings.
* Size: 195 rows with data from 32 individuals, 22 numerical features, 1 binary target variable 
* Instances: an 80/20 5-fold cross-validation train-test split was initially used (i.e., Tabular_Prototype_ZS.ipynb), but later models (i.e., outlined in Tabular_Aggregated.ipynb) were trained and evaluated using the Leave One Out method. 

#### Preprocessing / Clean up

Initial preprocessing (i.e., outlined in tabular_preprocessing.py) included removal of the ID column, class balancing through oversampling rows, and scaling data. Any outliers observed were kept in the data since there were relatively few data points and variations/anomalies could have been significant for diagnosis. For the second round of modelling, a different approach was used, and data was aggregated by patient ID. These data were then scaled prior to modelling as well.

#### Data Visualization

![Comparison of feature distributions between binary features during EDA.](/README_files/compare_binary.png)


![Baseline model confusion matrix showing overfitting.](/README_files/baseline_cmatrix.png)



### Problem Formulation

The inputs were the various voice and audio measures from voice recordings taken in the original patient study, which include frequency, amplitude, and pitch. These were used to determine whether an individual had Parkinson's or not.
  * Models used:
    * Decision Tree: primary/initial model used
    * K-Nearest Neighbor (KNN): a simpler model used after overfitting was observed in decision tree models
    * Random Forest: used for aggregated data to compensate for fewer data points
During each round of modelling, different feature selection interations were utilized. For the decision trees and KNNs, features were selected with the purpose of attempting to control multicollinearity. During the random forest/aggregated data phase, initial decision trees were used to determine the most influential features from each of the mean and variance datasets, and then concatenated to create a new "combo" dataset.
 

### Performance Comparison

During the first round of modelling, precision and ROC-AUC were used to evaluate model performance. However, likely a result of high multicolliinearity and data leakage from oversampled rows, early models had perfect precision and very high AUC (0.931). After hyperparameter tuning, this persisted, with a mean precision value of 0.945. Multiple KNN models were used as well and experienced similar, albeit less extreme, overfitting. This was despite heavy feature selection, which involved paring down the number of input features from 22 to 2-4. Multiple selections yielded similar results. In an attept to mitigate this, the data was then aggregated per patient, using mean and variance, and then modelled using Leave One Out train-test split. The perfromance of these models was evaluated through accuracy, the best model during this phase being Random Forest, with an accuracy of 0.844. It should be noted that this value may be inflated due to class imbalances, though the effects of this were attempted to be reduced through tuning the Random Forest model's class weights.




![Average AUC scores during first round of modelling after heavy feature selection; also shows overfitting (precision scores were perfect or near perfect during this phase).](/README_files/prototype_AUC.png)


![Model accuracy with aggregated data; different models were trained on mean data vs. variance data vs. a combination of the best features from both.](/README_files/aggregated_models.png)



### Conclusions

Although the initial decision trees used had very high precision and AUC scores, these values may have been inflated by data leakage and a result of overfitting. The random forest model trained on a combination of mean and variance data aggregated per patient is likely more applicable and generalizable despite having a lower accuracy score. Even then, it is difficult to compare these models one-to-one because of the variety of metrics used. 


### Future Work

Though options are limited due to the constrained nature/size of the dataset, in the future, I'd like to try better feature selection and engineering methods, to reduce the effects of multicollinearity, as well as use more complex modelling techniques on the aggregated version of the data.

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
* tabular_preprocessing.py: module containing functions for all preprocessing steps done in Tabular_Feasibility_ZS.ipynb
* Tabular_Prototype_ZS.ipynb: contains the first round of modelling
* Tabular_Aggregated.ipynb: contains the separate, final approach to modelling through aggregation of data by patient 

### Software Setup
All modelling and data manipulation was done using scikit-learn, pandas, and numpy. Majority of visualizations were completed with matplotlib.

### Data

The data can be downloaded on its [UCI webpage](https://archive.ics.uci.edu/dataset/174/parkinsons).

## Citations

Little, Max. "Parkinsons." UCI Machine Learning Repository, 2007, https://doi.org/10.24432/C59C74.





