# Credit Card risk Analysis with Machine Learning

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. It has been used in multiple fields and industries. For example, medical diagnosis, image processing, prediction, classification, learning association, regression etc. 

Machine Learning is divided into three types: supervised learning, unsupervised learning, and reinforcement learning. Although it may seem that the first refers to prediction with human intervention and the second does not, these two concepts are more related with what we want to do with the data.

## Overview

In this project we will apply machine learning to solve a real-world challenge: credit card risk. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we’ll need to employ different techniques to train and evaluate models with unbalanced classes. We are going to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card dataset, we’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we'll use a combinatorial approach of over and undersampling using the SMOTEENN algorithm. 

Next, we’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once we’re done, we’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk. Below are the list of Deliverables:

  * Deliverable 1: Resampling Models to Predict Credit Risk
  * Deliverable 2: The SMOTEENN Algorithm to Predict Credit Risk
  * Deliverable 3: Ensemble Classifiers to Predict Credit Risk
  * Deliverable 4: A Written Report on the Analysis (README.md)

## Analysis

We will use jupypter notebook with python packages for our Analysis. In all the steps we will import the dependencies, define the columns, Load the data. Then we will create the training and testing variables by converting the string values into numerical ones using the get_dummies() method. We can also create the target variables and check the balance of it using the below set of codes. 

### Creating target variables

![image](https://user-images.githubusercontent.com/85472349/137376774-34eddfae-944b-48f4-baf9-e858479c787f.png)


### Split the Data into Training and Testing


![image](https://user-images.githubusercontent.com/85472349/137376876-eaa7612a-b89e-4c0b-bece-d0ed7aad0bca.png)


### The precision, recall scores and finally F1 score 

After the accuracy score and with Confusion matrix values we will create a classification report. Basically the below methods are used to calculate the Precision, Recall and F1 score (harmonic mean):

![image](https://user-images.githubusercontent.com/85472349/137380263-ba50eca3-b6a9-414c-be76-26c7c909e163.png)

![image](https://user-images.githubusercontent.com/85472349/137380370-16a40359-7621-460f-ba23-b4f95733467f.png)

![image](https://user-images.githubusercontent.com/85472349/137380406-38c958b7-08c3-47f6-9e8d-b68852136ca4.png)

![image](https://user-images.githubusercontent.com/85472349/137380561-a594a6a5-4b75-41cd-9799-9b2ae4b112ba.png)


## Deliverable 1: Resampling Models to Predict Credit Risk

With our knowledge of the **imbalanced-learn and scikit-learn** libraries, we’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, we’ll use the oversampling **RandomOverSampler and SMOTE** algorithms, and then we’ll use the undersampling **ClusterCentroids** algorithm. Using these algorithms, we’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, a confusion matrix, and generate a classification report.


### Random Oversampling

![image](https://user-images.githubusercontent.com/85472349/137377083-56c5de65-5ca1-402d-8076-cc427b7f0f18.png)


### SMOTE Oversampling

![image](https://user-images.githubusercontent.com/85472349/137377270-dcfdb2c8-f825-4776-9a7f-374bbf82549f.png)


### ClusterCentroids Undersampling

![image](https://user-images.githubusercontent.com/85472349/137377356-fe2be66c-6a18-477c-abf4-552a9aa18e01.png)


## Deliverable 2: The SMOTEENN Algorithm to Predict Credit Risk

Using the **SMOTEENN algorithm,** we’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

### Combination (Over and Under) Sampling


![image](https://user-images.githubusercontent.com/85472349/137377884-17d9c757-0d9e-4f00-b7fb-84cd70163766.png)


## Deliverable 3: Ensemble Classifiers to Predict Credit Risk

With our knowledge of the **imblearn.ensemble** library, we’ll train and compare two different ensemble classifiers, **BalancedRandomForestClassifier and EasyEnsembleClassifier**, to predict credit risk and evaluate each model. Using both algorithms, we’ll resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

### Balanced Random Forest Classifier

![image](https://user-images.githubusercontent.com/85472349/137378539-a76ef9e4-ee1f-4a3e-bf50-c38985af4e4b.png)


#### Feature importance in descending order

![image](https://user-images.githubusercontent.com/85472349/137378669-e8eefc33-3c77-4fd5-a2d3-f31a868977f7.png)


### Easy Ensemble AdaBoost Classifier

![image](https://user-images.githubusercontent.com/85472349/137378726-4600071e-f3b2-4e4d-b678-7c5a9fdb559b.png)



## Results

So, from our above six Machine Learning Models we can describe the balanced accuracy scores, the precision, recall scores and finally F1 score (harmonic mean). Below are the corresponding values for each Algorithms: 

![image](https://user-images.githubusercontent.com/85472349/137380659-0401e660-0e05-4e67-8753-6169ac081331.png)


## Summary

From the above results and table of counts we can clearly see the balanced accuracy scores and the precision and recall scores and the F1 Score are high for **Easy Ensemble AdaBoost Classifier** Machine learning algorithm. Also, the F1 score can be characterized as a single summary statistic of precision and sensitivity(recall) and it is a great measurement to gauge model performance. From the above table values the best performing model is the Easy Ensemble Classifier with an average F1 score of 0.96. While the worst performing is Cluster Centroid with an average F1 score of 0.60. So, we can apply this machine learning technique to solve a **credit card risk in real-world.**
