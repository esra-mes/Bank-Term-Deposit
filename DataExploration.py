'''
This file was used to take an initial look at the DSA Data Set. It includes an exploration 
of each feature and was used to understand missing data as well. 
'''

#import packages needed
import pandas as pd
from scipy.stats import chi2_contingency 
from statistics import mean 

#read dataset
data = pd.read_csv('DSA Data Set.csv')

###Get basic information about data
numSamples = len(data)
data.head()
data.info()

print('-------------------------------------------------')

print(data.describe())


print('-------------------------------------------------')

#Group data by desired target (subscription to term deposit)
yGroup = data.groupby('y').size()
print(yGroup)

print('-------------------------------------------------')

#create a copy of the dataset and drop "duration" since we won't use in the model
df = data.copy()
df = df.drop('duration', axis=1)

#separate the categorical and numerical columns
catData = df.select_dtypes(exclude='number')
numData = df.select_dtypes(include='number')

#list of column names of categorical and numerical columns
catCols = catData.columns
numCols = numData.columns

#Get unique values in categorical columns 
def unique_values(data):
    for each in data:
        print(each, ":", data[each].unique())


#Get all categorical columns that contain 'unknown' (missing values)
unknownsList = []
def unknown_cols(data):
    for each in data:
        if 'unknown' in data[each].values:
            unknownsList.append(each) #added names of columns containing unknown to a list for later analysis
unique_values(catData)

print('-------------------------------------------------')
unknown_cols(catData)

print("Columns that contain 'unknown':", (unknownsList))
print('-------------------------------------------------')

#Check to see if there is a relationship between 'unknowns' and target

#use all columns with unknown and create dummy columns
unknowns_encoded = pd.get_dummies(df[unknownsList], drop_first=False)
unknown_cols = [col for col in unknowns_encoded.columns if 'unknown' in col]
df = pd.concat([df,unknowns_encoded], axis = 1)

#test if these columns have any relationship with y
def chiTest(colList, data, target):
    for each in colList:
        data_crosstab = pd.crosstab(data[each], target, margins = False)
        stat, p, dof, expected = chi2_contingency(data_crosstab)
        alpha = 0.05
        print(each, "p value is " + str(p)) 
        if p <= alpha: 
            print('Dependent (reject H0)') 
        else: 
            print('Independent (H0 holds true)')

results = chiTest(unknown_cols, unknowns_encoded, df['y'])
print(results)
print('-------------------------------------------------')
######Information for interpretation

duration = data['duration']
duration2  = [60 if n <= 60 else n for n in duration ]
meanDuraton = mean(duration2)
print('Average minutes on on call was ' + str(round(meanDuraton/60)))