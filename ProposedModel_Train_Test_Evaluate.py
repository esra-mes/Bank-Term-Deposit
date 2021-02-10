'''
This file was used to clean the data, fit a model (logistic regression) to training data, 
and test it on the  test data. The outputs are evaluation metrics such as a AUC, confusion matrix, 
classification_report, and an ROC curve to visualize AUC. 
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from imblearn import *
from imblearn.over_sampling import RandomOverSampler 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot


#read dataset
data = pd.read_csv('DSA Data Set.csv')
df = data.copy()
df = df.drop('duration', axis=1)

#separate the categorical and numerical columns
catData = df.select_dtypes(exclude='number')
numData = df.select_dtypes(include='number')

#list of column names of categorical and numerical columns
catCols = catData.columns
numCols = numData.columns


###########################Data Cleaning 

#use one-hot encoding to make features out of each category in categorical features
columns_encoded = pd.get_dummies(df[catCols], drop_first=False) #make dummy columns
df = pd.concat([df[numCols],columns_encoded], axis = 1) #concat new columns with the numerical columns only
df = df.drop(['y_no', 'ModelPrediction'], axis=1) #drop y_no because y is binary and ModelPrediction which wont be used in the new model
columns_All = list(df.columns) #sanity check

#split data and save the last 15% for final
#X is the array of features, y is target
X = df.drop(['y_yes'], axis=1) #df
y = df['y_yes'] #series

#simple split, will use training set in cross validation
X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)

###########################Train the model 

#oversample to deal with imbalanced data, oversample only training set
sampler = RandomOverSampler(random_state=42)
X_train, y_train = sampler.fit_resample(X_train1, y_train1)
pd.Series(y_train).value_counts().to_frame()

#define cross validation method, use stratified fold because data is imbalanced
cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

#define pipeline using standardScaler to normalize and LogReg as the algorithm
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
param_grid = {'logisticregression__C': [0.01, 0.01, 0.1, 1, 10, 100]} #different C or regularization values to test
grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring = 'roc_auc') #gridsearch using pipe line, grid, cv method mentioned above

scores = []
f1_scores= []

for train_ix, test_ix in cv.split(X_train, y_train):
	#select rows
    fold_X_train, fold_X_test = X_train.take(list(train_ix),axis=0), X_train.take(list(test_ix),axis=0)
    fold_y_train, fold_y_test = y_train.take(list(train_ix),axis=0), y_train.take(list(test_ix),axis=0)
    
    #fit the model using the training data and make predictions
    model_obj= grid.fit(fold_X_train, fold_y_train)
    fold_y_pred = model_obj.predict(fold_X_test)

    #score the model on the validation data and predictions it made
    score = accuracy_score(fold_y_test, fold_y_pred)
    f1 = f1_score(fold_y_test, fold_y_pred)
    report = classification_report(fold_y_test, fold_y_pred)
    conf_matrix = confusion_matrix(fold_y_test, fold_y_pred)

    #add scores to the list so that we can get the mean values 
    scores.append(score)
    f1_scores.append(f1)

#get the mean for each score type
mean_score = np.array(scores).mean()
mean_f1 = np.array(f1_scores).mean()

#print evaluation metrics
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))

#which model type was best out of those we tried (various C values in our case)
print("Best estimator:\n{}".format(grid.best_estimator_))

#mean accuracy and f1 scores of the model 
print('Accuracy scores of the model: {:.2f}'.format(mean_score))
print('F1 scores of the model: {:.2f}'.format(mean_f1))

print('\n Classification report of the model')
print('--------------------------------------')
print(report)

print('\n Confusion Matrix of the model')
print('--------------------------------------')
print(conf_matrix)

print('--------------------------------------')
print('--------------------------------------')
print('--------------------------------------')
print('Evaluating Model on Test Set')


########################### Test the model 

# create an object of the LinearRegression Model
logregmod = LogisticRegression(C=1, max_iter = 100000)

# fit the model with the training data
logregmod.fit(X_train1, y_train1)
print("Training set accuracy score: {:.3f}".format(logregmod.score(X_train1, y_train1)))

# making the predictions
predict_y_test  = logregmod.predict(X_test)
print("Test set score: {:.3f}".format(logregmod.score(X_test, y_test)))

# Getting the confusion matrix
confusion_matrix = confusion_matrix(y_test, predict_y_test)
print(confusion_matrix)

# getting the classification report
report = classification_report(y_test, predict_y_test)
print(report)

##### ROC Curve for the model

#no skill probabilities
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logregmod.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

#get coefficients for each feature
coefs = pd.DataFrame(
    logregmod.coef_[0] *
    X_train1.std(axis=0),
    columns=['Coefficients importance'], index=X.columns)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(coefs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()