#!/usr/bin/env python
# coding: utf-8

# ### Predicting Sales Opportunity Outcome

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import ydata_profiling
from ydata_profiling import ProfileReport

import warnings
warnings.filterwarnings("ignore")


# In[2]:


cars = pd.read_excel(r"C:\Users\jh095\OneDrive - Cummins\GAC\Project Management\data\cars.xlsx")
cars.head()


# In[3]:


cars.info()


# In[4]:


cars.describe()


# In[5]:


cars.columns


# In[6]:


cars['Total_Ratio'] = cars.apply(lambda x: x['Ratio_Identify'] + x['Ratio_Validate'] + x['Ratio_Qualify'], axis=1)


# In[7]:


cars['Total_Ratio'].describe()


# In[8]:


cars[cars['Total_Ratio'] > 1]


# In[9]:


cars[cars['Total_Ratio'] > 1].shape


# In[10]:


cars[cars['Total_Ratio'] > 1]


# In[11]:


# Only 1 entry with an extreme value.

cars[cars['Total_Ratio'] > 1.00001]


# In[12]:


# Drop the row with the extreme value.

cars = cars.drop(cars[cars['Total_Ratio'] > 1.00001].index)


# ### Rename values in categorical columns

# In[13]:


cars['Supplies'].replace(['Car Accessories', 'Performance & Non-auto', 'Tires & Wheels','Car Electronics'],
                         ['Accessories', 'Performance', 'Tires_Wheels', 'Electronics'], inplace=True)


# In[14]:


cars['Supplies_Sub'].replace(['Motorcycle Parts', 'Exterior Accessories', 'Garage & Car Care', 'Shelters & RV',
                               'Batteries & Accessories', 'Replacement Parts', 'Interior Accessories', 'Towing & Hitches',
                               'Performance Parts', 'Tires & Wheels', 'Car Electronics'],
                              ['Motorcycle_Parts', 'Exterior', 'Garage_Car_Care', 'Shelters_RV', 
                               'Batteries', 'Replacement', 'Interior', 'Towing_Hitches',
                               'Performance', 'Tires_Wheels', 'Electronics'], inplace=True)


# In[15]:


cars['Region'].replace('Mid-Atlantic', 'Mid_Atlantic', inplace=True)

cars['Market'].replace('Fields Sales', 'Field_Sales', inplace=True)


# In[16]:


cars['Client_Revenue'].replace(['100K or less', 'More than 1M', '250K to 500K', '500K to 1M', '100K to 250K'],
                               ['Below_100K', 'Above_1M', '250K_to_500K', '500K_to_1M', '100K_to_250K'], inplace=True)


# In[17]:


cars['Client_Employee'].replace(['1K or less', 'More than 25K', '5K to 15K', '1K to 5K', '15K to 25K'],
                                ['Below_1K', 'Above_25K', '5K_to_15K', '1K_to_5K', '15K_to_25K'], inplace=True)


# In[18]:


cars['Size'].replace(['40K to 50K', '10K to 20K', '30K to 40K', '10K or less', '20K to 30K',
                      '50K to 60K', 'More than 60K'],
                     ['40K_to_50K', '10K_to_20K', '30K_to_40K', 'Below_10K', '20K_to_30K',
                      '50K_to_60K', 'Above_60K'], inplace=True)


# In[19]:


# 'ID' column is the original index, which is not needed.
# 'Total_Ratio' column was used as a quality check and is no longer needed.

cars.drop(['ID', 'Total_Ratio'], axis=1, inplace=True)


# # Exploratory Data Analysis

# In[20]:


num_columns = ['Amount', 'Elapsed_Days', 'Stage_Change', 'Total_Days', 'Total_Siebel',
              'Ratio_Identify', 'Ratio_Validate', 'Ratio_Qualify']


# In[21]:


cat_columns = ['Result', 'Supplies_Sub', 'Region', 'Market', 'Client_Revenue', 
               'Client_Employee', 'Client_Past', 'Competitor', 'Size']


# In[22]:


# Correlation between numeric columns.

cars_corr = cars.corr()

plt.subplots(figsize=(12,9))
heatmap = sns.heatmap(cars_corr, annot=True)


# In[23]:


# Plot the distribution of numerical columns.

fig, axes = plt.subplots(4,2, figsize=(14,12))
sns.distplot(cars['Amount'], color='blue', ax=axes[0,0])
sns.distplot(cars['Elapsed_Days'], color='red', ax=axes[0,1])
sns.distplot(cars['Total_Days'], color='gold', ax=axes[1,0])
sns.distplot(cars['Total_Siebel'], color='teal', ax=axes[1,1])
sns.distplot(cars['Stage_Change'], color='orange', ax=axes[2,0])
sns.distplot(cars['Ratio_Identify'], color='green', ax=axes[2,1])
sns.distplot(cars['Ratio_Validate'], color='purple', ax=axes[3,0])
sns.distplot(cars['Ratio_Qualify'], color='cyan', ax=axes[3,1])
plt.tight_layout()


# In[24]:


# Plot the distribution of categorical columns.

fig, axes = plt.subplots(5,2, figsize=(14,20))
sns.countplot(y=cars['Result'], ax=axes[0,0])
sns.countplot(y=cars['Supplies_Sub'], ax=axes[0,1])
sns.countplot(y=cars['Region'], ax=axes[1,0])
sns.countplot(y=cars['Market'], ax=axes[1,1])
sns.countplot(y=cars['Client_Revenue'], ax=axes[2,0])
sns.countplot(y=cars['Client_Employee'], ax=axes[2,1])
sns.countplot(y=cars['Client_Past'], ax=axes[3,0])
sns.countplot(y=cars['Competitor'], ax=axes[3,1])
sns.countplot(y=cars['Size'], ax=axes[4,0])
fig.delaxes(axes[4][1])
plt.tight_layout()


# In[25]:


#%config InlineBackend.figure_format = 'retina'
#sns.set_style('whitegrid')


# ## Feature Engineering

# In[26]:


# Change values of 'Result' column to binary.
# Reduce the number of categories in 'Client_Past' column to 2 (0 = new client, 1 = existing client).

cars['Result'] = cars['Result'].map(lambda x: 0 if x == 'Loss' else 1)
cars['Client_Past'] = cars['Client_Past'].map(lambda x: 0 if x == '0 (No business)' else 1)


# In[27]:


# 'Supplies_Sub' column provides a clearer picture compared to 'Supplies' column

cars.groupby('Supplies').Supplies_Sub.value_counts()


# In[28]:


# Drop 'Total_Siebel' column due to high correlation with 'Total_Days' column.
# Drop 'Supplies' column since 'Supplies_Sub' is more detailed.

cars2 = cars.drop(['Total_Siebel', 'Supplies'], axis=1)


# In[29]:


# Create dummy variables for categorical columns.

cat_dummy = pd.get_dummies(cars2[cat_columns], drop_first=True)
print(cat_dummy.shape)
cat_dummy.head()


# In[30]:


# Combine original dataset with new dummy variables.
# Drop old categorical columns.

print(cars2.shape)
cars2.drop(cat_columns, axis=1, inplace=True)
print(cars2.shape)
cars2 = pd.concat([cars2, cat_dummy], axis=1)
print(cars2.shape)
cars2.head()


# In[31]:


cars2.head()


# #### Prepare target/predictor variables and train/test sets

# To prepare our data, we will split it into two parts: a training set and a testing set. We will use the training set to train our models, and we will use the testing set to evaluate how well our models perform on unseen data. We will also use a technique called 5-fold cross validation to train our models.
# 
# We noticed that the target column has an imbalance, with the majority class being overrepresented. This is a concern, and we will address it in the next section.
# 
# Our models will need to achieve an accuracy score that is higher than the baseline of 0.77408

# In[32]:


# Set target y and predictor X.
# Calculate overall baseline accuracy.

y = cars2['Result']
X = cars2.drop('Result', axis=1)

print(y.value_counts())

baseline = 1 - np.mean(y)
baseline


# In[33]:


# Split train/test sets.

from sklearn.model_selection import cross_val_score, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)
print('Training data:')
print(X_train.shape)
print(y_train.shape)
print('------------------')
print('Testing data:')
print(X_test.shape)
print(y_test.shape)


# In[34]:


# Calculate baseline accuracy for both training and testing sets.

print('Training set:')
print(y_train.value_counts())
print()
print('Training baseline accuracy: ' + str(1 - np.mean(y_train)))
print('-------------------')
print('Testing set:')
print(y_test.value_counts())
print()
print('Testing baseline accuracy: ' + str(1 - np.mean(y_test)))


# In[35]:


# Standardize X.

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xs_train = ss.fit_transform(X_train)
Xs_test = ss.transform(X_test)


# Next, we will use the SelectFromModel function from scikit-learn to perform feature selection. This function will use a Random Forest Classifier to identify the features that are most important for predicting the opportunity outcome.
# 
# The results show that only 9 of the 44 features in our dataset were selected. This means that these 9 features are the most important for predicting the opportunity outcome.

# In[36]:


# Perform feature selection.

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(RandomForestClassifier())

print('Shape of dataset before feature selection: ' + str(X_train.shape))
select.fit(Xs_train, y_train)
Xs_train = select.transform(Xs_train)
Xs_test = select.transform(Xs_test)
print('Shape of dataset after feature selection: ' + str(Xs_train.shape))


# In[37]:


# Check with features were selected.

feature_support = pd.DataFrame({'feature': X_train.columns,
                           'support': select.get_support()})

feature_support.sort_values('support', inplace=True, ascending=False)
feature_support.head(10)


# #### Data Imbalance

# In order to address the issue of class imbalance, we intend to employ two distinct techniques provided by the Python package 'imblearn' to rebalance the data in our training dataset.
# 
# RandomOverSampler: This technique involves randomly oversampling the minority class (1) with replacement. Through this process, we will augment the number of instances in the minority class to reach 42,215 from the original count of 12,401.
# 
# RandomUnderSampler: In contrast, this method randomly undersamples the majority class (0) with replacement. Consequently, the number of majority class instances will be reduced to 12,401 from the initial count of 42,215.
# 
# By applying these techniques, we will generate three distinct training datasets: the default dataset, the oversampled dataset, and the undersampled dataset

# In[38]:


from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=100)
Xs_train_rus, y_train_rus = rus.fit_resample(Xs_train, y_train)

print('Dataset before undersampling:')
print(y_train.value_counts())
print('------------------------------')
print('Dataset after undersampling:')
print(Xs_train_rus.shape)
print(y_train_rus.shape)


# #### Model 9 K-Nearest Neighbours with Undersampling

# In[42]:


# Import necessary libraries
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

k_range = range(1, 21)
knn_params = {'n_neighbors': k_range, 'weights': ['uniform', 'distance']}


# In[43]:


# Perform Grid Search with undersampled dataset to identify best parameters.

knn_params = {'n_neighbors':k_range,
              'weights':['uniform','distance']}

knn_rus_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, verbose=1, n_jobs=-1)
knn_rus_grid.fit(Xs_train_rus, y_train_rus)


# In[44]:


# View grid search results.

print(knn_rus_grid.best_score_)
print(knn_rus_grid.best_params_)
print(knn_rus_grid.best_estimator_)


# In[54]:


knn9_rus = KNeighborsClassifier(n_neighbors=15, weights='distance')

knn9_rus.fit(Xs_train_rus, y_train_rus)
knn9_rus_train_score = knn9_rus.score(Xs_train_rus, y_train_rus)
knn9_rus_cv_score = np.mean(cross_val_score(knn9_rus, Xs_train_rus, y_train_rus, cv=5))
knn9_rus_test_score = knn9_rus.score(Xs_test, y_test)
y9_pred_rus = knn9_rus.predict(Xs_test)

y_test.to_csv(r'C:\Users\jh095\OneDrive - Cummins\GAC\Project Management\test.csv', index=False)

y9_proba = knn9_rus.predict_proba(Xs_test)
y9_proba = [p[1] for p in y9_proba]
knn9_roc_score = roc_auc_score(y_test, y9_proba)

print('training score:', knn9_rus_train_score)
print('training CV score:', knn9_rus_cv_score)
print('testing score:', knn9_rus_test_score)
print('ROC score:', knn9_roc_score)
print('------------------------------------------------------')
print(classification_report(y_test, y9_pred_rus))
print('------------------------------------------------------')
print(y_test, y9_pred_rus)


# In[46]:


import pickle

path = r"C:\Users\jh095\OneDrive - Cummins\GAC\Project Management\code\deployment\model_dep.pkl"

# create an iterator object with write permission - model.pkl

with open(path, 'wb') as files:
    pickle.dump(knn9_rus, files)


# In[ ]:


# import streamlit as st
# import joblib


# In[ ]:


# model = joblib.load(r'C:\Users\jh095\OneDrive - Cummins\GAC\Project Management\code\deployment\model.pkl')

 

# st.title('Sales Opportunity Prediction Deployment')
# st.write('This is a Streamlit web app to demonstrate deploying a Sales Opportunity Prediction model.')


# In[53]:


import joblib

# Save your trained scikit-learn model
joblib.dump(knn9_rus, r"C:\Users\jh095\OneDrive - Cummins\GAC\Project Management\code\deployment\model_dep2.pkl")


# In[55]:


# Load your trained scikit-learn model
model = joblib.load(r"C:\Users\jh095\OneDrive - Cummins\GAC\Project Management\code\deployment\model_dep2.pkl")

def make_predictions(input_csv):
    # Read the input data from a CSV file
    data = pd.read_csv(r'C:\Users\jh095\OneDrive - Cummins\GAC\Project Management\test.csv')

    # Make predictions
    predictions = model.predict(data)  # Replace 'data' with the actual input data

    return predictions


# In[ ]:




