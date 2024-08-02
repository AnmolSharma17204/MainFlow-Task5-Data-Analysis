'''
Task 5
Description:
Engineer new features and select relevant
features for model training.
Responsibility:
1.Generate meaningful features from existing
data.
2.Use techniques like PCA or feature
importance to select the most important
features.
Optimize feature sets for improved model
performance.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("D:\Python\Datasets\heart.csv")
#print(df.head(5))

#Checking columns names
print(df.columns.values)

#Checking for null values
print(df.isna().sum())

#plotting Histogram

# df.hist(bins=100,grid=False,figsize=(20,16))
# plt.hist(df,bins=50)
# plt.show()
print(df.describe())

# Ques. 1 How many persons have heart disease and how many people do not have heart disease

df.target.value_counts()\
#plotting results using bar chart
df.target.value_counts().plot(kind='bar',color='red',alpha=0.7)
plt.title("Heart Disease",fontsize=25)
plt.xlabel("0=No Disease 1=Disease",fontsize=15)
plt.ylabel("Amount",fontsize=15)
plt.show()

#using pie chart
df.target.value_counts().plot(kind='pie',autopct='%1.0f%%')
plt.legend(['No Disease','Disease'])
plt.show()


# How many male and female are in dataset
df.sex.value_counts()
#Plotting results
df.sex.value_counts().plot(kind='pie',autopct='%1.0f%%')
plt.legend(['Female','Male'])
plt.legend("Male Female Ratio")
plt.show()

#Ques.2 People of which sex has the most heart disease
pd.crosstab(df.target,df.sex)
sns.countplot(x='target',hue='sex',data=df)
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0=No Disease 1=Disease")
plt.ylabel("Amount")
plt.show()


#Ques.3 People of which sex has which type of chest pain most?
cp=df.cp.value_counts()
print(cp)
#plotting results
# df.cp.value_counts().plot(kind='pie',autopct='%1.0f%%')
# plt.legend(['ASY','ATA','NAP','TA'])
# plt.title("Chest Pain")
# plt.show()  
df.cp.value_counts().plot(kind='bar',color=['r','g','b','y'],alpha=0.7)
plt.title("Chest Pain Type vs count")

pd.crosstab(df.sex,df.cp)
#plotting crosstab
pd.crosstab(df.sex,df.cp).plot(kind='bar',color=['r','g','b','y'],alpha=0.7)
plt.title("Type of chest pain for sex")
plt.xlabel('0=Female 1=Male')
plt.ylabel('Amount')
plt.show()

#Ques.4 People with which chest pain are most pron to have heart disease?
pd.crosstab(df.target,df.cp)

sns.countplot(x=df.cp,data=df,hue='target')

#Plotting for age
sns.displot(x='age',data=df,bins=30,kde=True)

#plotting for maximum heart rate
sns.displot(x='thalach',data=df,bins=30,kde=True)

 
 #Ques. 5 How Many person having fbs value 0 or  1 are pron to have heart disease?
fbs=pd.crosstab(df.fbs,df.target).plot(kind='bar')
print(fbs)
pd.crosstab(df.target,df.fbs).value_counts()
plt.xlabel("Fbs values")
plt.ylabel("Affected persons")
plt.title("fbs values and affected persons")
plt.legend(['Not Affected','Affected'])


#Ques.6 Number of persons prone to heart disease on the basis of exang
fbs=pd.crosstab(df.exang,df.target)
print(fbs)  
pd.crosstab(df.exang,df.target).plot(kind='bar')

#Ques. 7 ca values and number of affected persons
print(pd.crosstab(df.ca,df.target))
pd.crosstab(df.ca,df.target).plot(kind='bar')
plt.ylabel("Affected persons")
plt.xlabel("ca values")
plt.title("ca values and number of  affected persons") 
plt.legend(['Not Affected','Affected'])