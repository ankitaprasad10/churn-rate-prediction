
# coding: utf-8

# # Churn Rate Analysis using Artificial Neural Network
# Customer churn is defined as the loss of customers because they move out to competitors. 
# It is an expensive problem in many industries since acquiring new customers costs five to six times more than retaining existing ones.
# Through our project we are predicting the number of customers who might leave the bank in furture.

# #In our project we have analysed 10,000 bank customers and predicted the Churn rate.
# We have used Churn_modelling.csv from Kaggle.com.
# 
# 
# # Key features of the data -
# We are using Churn_Modelling.csv from Kaggle.
# The data consists of 12 independent variables and one dependent variable.
# The 12 independent variable give us the information about customers and the dependent variable ‘Exited’ suggests us whether the customer is leaving the bank or not.
# The independent variables which will considered to predict the churn rate will be – CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading the data set
dataset = pd.read_csv('C:\\Users\\ankit\\Downloads\\Churn_Modelling.csv')


# In[4]:


dataset


# In[5]:


dataset.head()


# In[6]:


dataset.shape


# In[7]:


#Checking for null values in the data set


# In[8]:


dataset.isnull().any()


# In[9]:


#Creating dummy matrix for Geography column


# In[10]:


dummies = pd.get_dummies(dataset['Geography']) 
dummies = dummies.add_prefix("{}#".format('Geography'))


# In[11]:


dummies


# In[12]:


#Dropping the original Geography column


# In[13]:


dataset.drop('Geography', axis=1, inplace=True)
#dataset = dataset.join(dummies)
#dataset
dataset=pd.concat([dummies, dataset], axis=1)


# In[14]:


#Updated dataset


# In[15]:


dataset


# In[16]:


#Creating Dummy columns for gender column


# In[17]:


dummies_1 = pd.get_dummies(dataset['Gender'])
dummies_1


# In[18]:


dataset.drop('Gender', axis=1, inplace=True)


# In[19]:


dataset=pd.concat([dummies_1, dataset], axis=1)


# In[20]:


#Updated data set after dropping original Gender Column and concatinating the dummy gender column


# In[21]:


dataset


# In[22]:


#Dropping the extra columns


# In[23]:


dataset.drop('Geography#France', axis=1, inplace=True)


# In[24]:


dataset.drop('Male', axis=1, inplace=True)


# In[25]:


dataset.drop('CustomerId', axis=1, inplace=True)


# In[26]:


dataset.drop('RowNumber', axis=1, inplace=True)
dataset.drop('Surname', axis=1, inplace=True)


# In[27]:


#Final Updated Dataset


# In[28]:


dataset


# In[29]:


#Fetching values of relevant sttributes for which values in Exited column is 1


# In[30]:


avgSalaryForLeavingCustomers=0.0
avgSalaryForRetainedCustomers=0.0
count=0
count1=0
EstSalary=[]
EstSalary1=[]
AgeGroup1=[]
AgeGroup2=[]
HasCrCardGr1=[]
HasCrCardGr2=[]
creditScores1=[]
creditScores2=[]
balance1=[]
balance2=[]
noOfProductsGroup1=[]
noOfProductsGroup2=[]
tenureGroup1=[]
tenureGroup2=[]
for i in range(len(dataset)):
    if dataset['Exited'].loc[i]==1:
        count=count+1;
        EstSalary.append(dataset['EstimatedSalary'].loc[i])
        creditScores1.append(dataset['CreditScore'].loc[i])
        AgeGroup1.append(dataset['Age'].loc[i])
        HasCrCardGr1.append(dataset['HasCrCard'].loc[i])
        noOfProductsGroup1.append(dataset['NumOfProducts'].loc[i])
        balance1.append(dataset['Balance'].loc[i])
        avgSalaryForLeavingCustomers= dataset['EstimatedSalary'].loc[i]+avgSalaryForLeavingCustomers
    elif dataset['Exited'].loc[i]==0:
        count1=count1+1;
        EstSalary1.append(dataset['EstimatedSalary'].loc[i])
        creditScores2.append(dataset['CreditScore'].loc[i])
        HasCrCardGr2.append(dataset['HasCrCard'].loc[i])
        AgeGroup2.append(dataset['Age'].loc[i])
        balance2.append(dataset['Balance'].loc[i])
        noOfProductsGroup2.append(dataset['NumOfProducts'].loc[i])
        avgSalaryForLeavingCustomers= dataset['EstimatedSalary'].loc[i]+avgSalaryForLeavingCustomers
        
print(count)
print(count1)


# In[31]:


#Plotting graph between relevant attributes and exited column to analyse which group is exiting the most in each attributes.


# In[32]:


#Histogram between 'has credit card' and Exited = 1


# In[33]:


import matplotlib.pyplot as plt
n_bins = 10
plt.hist(HasCrCardGr1, n_bins, histtype='bar', color='red',stacked='true',rwidth=0.5)


# In[34]:


#Histogram between AgeGroup and Exited


# In[35]:


import matplotlib.pyplot as plt
n_bins = 10
plt.hist(AgeGroup1, n_bins,histtype='bar', color='red',stacked='true',rwidth=0.5)
##plt.hist(EstSalary,num_binsalpha=0.5)


# In[36]:


print(avgSalaryForLeavingCustomers/2037)


# In[37]:


avgSalaryForRetainedCustomers=0.0
count=0
for i in range(len(dataset)):
    if dataset['Exited'].loc[i]==0:
        count=count+1;
        avgSalaryForRetainedCustomers= dataset['EstimatedSalary'].loc[i]+avgSalaryForRetainedCustomers
print(count)


# In[38]:


print(avgSalaryForRetainedCustomers/7963)


# In[39]:


#Histogram between Credit Score and Exited


# In[40]:


import matplotlib.pyplot as plt
n_bins = 10
plt.hist(creditScores1, n_bins, histtype='bar', color='red',stacked='true',rwidth=0.5) 
plt.show()


# In[41]:


#Histogram between balance and Exited


# In[42]:


import matplotlib.pyplot as plt
n_bins = 10
plt.hist(balance1, n_bins, histtype='bar', color='red',stacked='true',rwidth=0.5)


# In[43]:


#Histogram between Number of products owned by the customer and Exited


# In[44]:


import matplotlib.pyplot as plt
n_bins = 10
plt.hist(noOfProductsGroup1, n_bins, histtype='bar', color='red',stacked='true',rwidth=0.5)


# In[45]:


dataset.shape


# In[46]:


dataset


# In[47]:


#Assigning independent and dependent values from data set in variables


# In[48]:


X = dataset.iloc[:, :11].values
y = dataset.iloc[:, 11].values


# In[49]:


X.shape


# In[50]:


y.shape


# In[51]:


X,y


# In[52]:


#Creating test and training set. Splitting data into 80:20 ratio for training and test respectively.


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[54]:


X_train, X_test, y_train, y_test


# In[55]:


X_test.shape,X_train.shape,y_train.shape,y_test.shape


# In[56]:


#Standardizing the variables to bring them all to the same scale


# In[57]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[58]:


from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression()
logisticRegr.fit(X=X_train, y=y_train)


# In[59]:


y_pred=logisticRegr.predict(X_test)


# In[60]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


# In[61]:


# create heatmap
import seaborn as sns
fig, ax = plt.subplots()

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[62]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[63]:


from sklearn.naive_bayes import GaussianNB

model=GaussianNB()
model.fit(X=X_train, y=y_train)

y_pred=model.predict(X_test)


# In[64]:


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


# In[65]:


# create heatmap
fig, ax = plt.subplots()

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[66]:


#Importing Deep learning libraries


# In[67]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[68]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[69]:


# Initialising the ANN
classifier = Sequential()


# In[60]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))


# In[61]:


# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))


# In[64]:


classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))


# In[65]:


classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))


# In[66]:


# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[67]:



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[68]:


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# In[69]:


#Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[70]:


y_pred


# In[71]:


y_pred = (y_pred > 0.5)


# In[72]:


y_pred


# In[73]:


y_p = (y_pred==True)


# In[74]:


y_p


# In[75]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[76]:


cm


# In[77]:


accuracy=(1482+229)/2000


# In[78]:


accuracy


# # Conclusion and Future Scope

# Our model gives around 85% overall accuracy for the prediction of the customer churn.
# 
# For future work, several issues can be considered:
# 
# First, as the data pre-processing stage in data mining is a very important step for the final prediction model performance, the dimensionality reduction or feature selection step can be involved in addition to data reduction. 
# 
# Second, along with neural networks, other popular prediction techniques can be applied in combination, such as support vector machines, genetic algorithms, etc to develop hybrid models. 
# 
# Finally, the current methodology of churn prediction can be tested for other sectors like banking, insurance or air line and comparisons can be done for prediction accuracy.
