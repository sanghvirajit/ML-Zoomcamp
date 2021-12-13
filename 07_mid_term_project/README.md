# Auto Insurance Fraud Detection

## Business case

Claim related fraud is a huge problem in the insurance industry. It is quite complex and difficult to identify those unwanted claims. With Machine Learning Algorithm, I am trying to troubleshoot and help the General Insurance industry with this problem.

The data that I have is from Automobile Insurance. I will be creating a predictive model that predicts if an insurance claim is fraudulent or not. The answere between YES/NO, is a Binary Classification task.

A comparison study has been performed to understand which ML algorithm suits best to the dataset.

The raw dataset is in the file **"insurance_claim.csv"** which contains 1000 rows and 39 columns.

## Steps Involved

**Part 1** - Data Processing : Cleaning and Transforming Raw Data into the Understandable Format

**Part 2** - Profiling : Data profiling is the process of examining the data available from an existing information source (e.g. a database or a file) and collecting statistics or informative summaries about that data.

**Part 3** - Exploratory Data Analysis - EDA: Finding insights from the Data

**Part 4** - **Logistic regression, Decision Trees, and KNN**: Building the models and hyperparameter tuning, to find the best parameters.

**Part 5** - **Ensemble Learning and Random Forest**: Random forest and hyperparameter tuning, to find the best parameters.

**Part 6** - **Gradient Boosting and XGBoost**: XGBoost and hyperparameter tuning, to find the best parameters.

**Part 7** - **LightGBM and CatBoost**: Hyperparameter tuning, to find the best parameters.

**Part 7** - War between Boosting algorithm, Selecting the best model: **XGBoost gives the best results on test dataset**.

**Part 8** - Deploying the model on AWS Cloud

## Over-Sampling with SMOTE

One approach to addressing imbalanced datasets is to oversample the minority class. 
The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. 
Instead, new examples can be synthesized from the existing examples. 
This is a type of data augmentation for the minority class and is referred to as the **Synthetic Minority Oversampling Technique, or SMOTE** for short.

### Before SMOTE

![imbalance](https://user-images.githubusercontent.com/69073063/139042277-cfe8f2c5-7e13-4005-bebc-4359a1e9c2dd.png)

### After SMOTE

![balance](https://user-images.githubusercontent.com/69073063/139042373-cc1b2cda-c426-4f05-8643-cd43ea39a593.png)

## Model Summary

![model_summary](https://user-images.githubusercontent.com/69073063/138944750-e64b36dd-d05c-4fb7-9e6f-fbac7e466a53.png)

### ROC_AUC Curve of the final model on **test dataset**

![ROC_AUC_CURVE](https://user-images.githubusercontent.com/69073063/139042538-4933927c-1620-481f-aa20-655a01caaeb3.png)

## Requirements

```scala
Python 3.8
numpy: 1.21.2
pandas: 1.3.4
scikit-learn: 1.0
scipy: 1.7.1
waitress: 2.0.0
flask: 2.0.2
xgboost: 1.5.0
requests: 2.26.0
awsebcli: 3.20.2
``` 

## Commands to run the project locally
**make sure your docker daemon is working**

```scala
git clone https://github.com/sanghvirajit/mid-term-project.git
activate conda py38
python3 -m pipenv --python 3.8 shell 
docker build -t fraud_detection .
docker run -it --rm -p 9696:9696 fraud_detection
python3 predict-test-docker.py
``` 

## Commands to run the project from cloud service

**Service is already running on the cloud.**

![AWS](https://user-images.githubusercontent.com/69073063/139234840-b3846b5c-ec37-47f5-bd58-e26e3f942ff5.png)

**To test it, just run the following command in console.**

```scala
python3 predict-test-cloud.py
``` 

![test-result](https://user-images.githubusercontent.com/69073063/139287373-ce68d061-1381-4291-8969-912e099ddce3.png)

**I will terminate the service on 09.11.2021, after the end of peer-review week**

**Commands used to deploy the model on cloud, You do not need to run the following commands**

```scala
git clone https://github.com/sanghvirajit/mid-term-project.git
activate conda py38
python3 -m pipenv --python 3.8 shell 
pipenv install awsebcli –dev
eb init –p docker –r eu-central-1 fraud_detection 
eb local run 
eb create fraud-detection-env
``` 

**public endpoint that could be tested:** 

```scala
python3 predict-test-cloud.py
``` 


# LightGBM vs. XGBoost vs. CatBoost 

XGBoost was originally produced by University of Washington researchers and is maintained by open-source contributors. XGBoost uses the gradients of different cuts to select the next cut, but XGBoost also uses the hessian, or second derivative, in its ranking of cuts. Computing this next derivative comes at a slight cost, but it also allows a greater estimation of the cut to use.

LightGBM is a boosting technique and framework developed by Microsoft. LightGBM is unique in that it can construct trees using Gradient-Based One-Sided Sampling, or GOSS for short. GOSS looks at the gradients of different cuts affecting a loss function and updates an underfit tree according to a selection of the largest gradients and randomly sampled small gradients. GOSS allows LightGBM to quickly find the most influential cuts.

CatBoost is developed and maintained by the Russian search engine Yandex. CatBoost distinguishes itself from LightGBM and XGBoost by focusing on optimizing decision trees for categorical variables, or variables whose different values may have no relation with each other (eg. apples and oranges). To compare apples and oranges in XGBoost, you’d have to split them into two one-hot encoded variables representing “is apple” and “is orange,” but CatBoost determines different categories automatically with no need for preprocessing (LightGBM does support categories, but has more limitations than CatBoost).

# My own findings

Catboost outperformed LightGBM and gave very similar finding when compared to XGBoost, with XGBoost resulting validation auc slightly better than CatBoost.

Training with CatBoost was much faster than XGBoost.

My dataset was small, hence I finally opted for XGBoost, but if incase my dataset were large enough, I would have definitely opted for CatBoost with SMOTE as my final model.
