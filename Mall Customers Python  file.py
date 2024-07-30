import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#loading the dataset
df = pd.read_csv('Mall_Customers.csv')
#Checking for features in the dataset
features = df.columns
print(df.info())
print(features)
#Checking for  missing values
missing_values = df.isnull().sum()
#print(missing_values)

#How does the 'Annual income', 'Age' and 'Spending Score' vary with 'Gender'? Use graphs as part of your description.[9]
#Barplot
sns.barplot(df, x='Gender', y='Annual Income (k$)')
plt.xlabel('Gender')
plt.ylabel('Annual Income (k$)')
plt.title('Annual Income by gender')
plt.show()
#boxlpot
sns.boxplot(df, x='Gender', y='Spending Score (1-100)')
plt.xlabel('Gender')
plt.ylabel('Spending Score (1-100)')
plt.title('Spending Score (1-100) by gender')
plt.show()
#violin
sns.violinplot(df, x='Gender', y='Age')
plt.title('Age Distribution by Gender')
plt.show()
#Using the KMeans method to place the data in groups (clusters) based on customer 'Age' and 'Annual Income'
from sklearn.cluster import KMeans

#Targeting my specified columns
df = pd.read_csv('Mall_Customers.csv', usecols=['Age', 'Annual Income (k$)'])

x_train, x_test = train_test_split(df[['Age', 'Annual Income (k$)']], test_size=0.33, random_state=0)
# Normalizing the training data
x_train_norm = normalize(x_train)

# Normalizing the testing data 
x_test_norm = normalize(x_test)

# Creating a KMeans model
model = KMeans(n_clusters=3, random_state=0, n_init='auto')

# Fiting the model to the normalized training data
model.fit(x_train_norm)

#Calculating the silhouette score to assess how well data points are separated within clusters
silhouette = silhouette_score(x_train_norm, model.labels_, metric='euclidean')
print(f"Silhouette score: {silhouette}")

#Visualize the cluster distribution 
sns.scatterplot(data=x_train, x='Age', y='Annual Income (k$)', hue=model.labels_)
plt.show()

perf = silhouette_score(x_train_norm, model.labels_, metric='euclidean')
print(perf)

#testing cluster range
K = range(10, 20)
fits=[]
score=[]
for k in K:
#train the model for the current value of k on training data
    model = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(x_train_norm)
#append the models to fit
    fits.append(model)
#append the silouette_score
    score.append(silhouette_score(x_train_norm, model.labels_, metric='euclidean'))

print(fits)
print(score)
#visualize a few, start with k=0
sns.scatterplot(data=x_train, x='Age', y='Annual Income (k$)', hue=fits[0].labels_)
plt.show()

#visualize a few, start with k=2
sns.scatterplot(data=x_train, x='Age', y='Annual Income (k$)', hue=fits[2].labels_)
plt.show()

#visualize a few, start with k=3
sns.scatterplot(data=x_train, x='Age', y='Annual Income (k$)', hue=fits[3].labels_)
plt.show()


