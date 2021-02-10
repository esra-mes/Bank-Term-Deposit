'''
This file was used to evaluate the model currently used by the bank.
The output are evalution metrics (accuracy, recall, f1, precision), 
a confusion matrix, as well as the ROC curve'''


import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


#load data, make copy, subset y-actual (y) and y-prediction (ModelPrediction)
data = pd.read_csv('DSA Data Set.csv')
df = data.copy()
df = df[['y','ModelPrediction']]

#convert values to 0's (No), and 1's (Yes)
#use 0.5 threshold to evaluate model prediction initially
df['ModelPrediction']  = [1 if n >= 0.5 else 0 for n in df['ModelPrediction'] ]
df['y']  = [1 if n == 'yes' else 0 for n in df['y'] ]
y_true = df['y']
y_pred = df['ModelPrediction']

#sanity check
#print(df.head())
#print(df['ModelPrediction'].unique())
#print(df['y'].unique())

#create matrix to see TP,TF,FN,FP
confusion_matrix = pd.crosstab(df['y'], df['ModelPrediction'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

#calculate and print different evaluation scores
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print('--------------------------------------')
print('The accuracy of this model at the 0.5 cut off is', + accuracy)
print('--------------------------------------')
print('The precision of this model at the 0.5 cut off is', + precision)
print('--------------------------------------')
print('The recall of this model at the 0.5 cut off is', + recall)
print('--------------------------------------')
print('The f1 score of this model at the 0.5 cut off is', + f1)
print('--------------------------------------')
print(classification_report(y_true, y_pred))
print('--------------------------------------')

##################
##################
#AUC and ROC curves 

#subset data
auc_data = data.copy()
y_pred_auc = auc_data['ModelPrediction']

##### ROC Curve for the model

#no skill probabilities
ns_probs = [0 for _ in range(len(y_true))]

# calculate scores
ns_auc = roc_auc_score(y_true, ns_probs)
lr_auc = roc_auc_score(y_true, y_pred_auc)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred_auc)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel("False Positive Rate // False Alarm Rate // Call, Not Interested")
pyplot.ylabel('True Positive Rate // Sensitivity // Recall // Call, Interested')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()