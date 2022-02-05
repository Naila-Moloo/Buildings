import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from matplotlib import pyplot

# Load dataset
df = pd.read_csv("/Users/nailamolooicloud.com/Downloads/Mall_Customers.csv")
del df['CustomerID']
print(df.head)

sns.countplot(df['Gender'])

gender= {'Male':0, 'Female':1}
df['Gender']= df['Gender'].map(gender)
print(df.head)

plt.figure(figsize=(10,6))
plt.scatter(df['Age'],df['Annual Income (k$)'], marker='o');
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Scatter plot between Age and Annual Income')

plt.figure(figsize=(10,6))
plt.scatter(df['Age'],df['Spending Score (1-100)'], marker='o');
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatter plot between Age and Spending Score')

plt.figure(figsize=(10,6))
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'], marker='o');
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatter plot between Annual Income and Spending Score')

plt.figure(figsize=(10,6))
plt.scatter(df['Gender'],df['Annual Income (k$)'], marker='o');
plt.xlabel('Gender')
plt.ylabel('Annual Income (k$)')
plt.title('Scatter plot between Gender and Annual Income')

plt.figure(figsize=(10,6))
plt.scatter(df['Gender'],df['Spending Score (1-100)'], marker='o');
plt.xlabel('Gender')
plt.ylabel('Spending Score (1-100)')
plt.title('Scatter plot between Gender and Spending Score')

plt.figure(figsize=(10,6))
plt.scatter(df['Gender'],df['Age'], marker='o');
plt.xlabel('Gender')
plt.ylabel('Age')
plt.title('Scatter plot between Gender and Age')

fig_dims = (10, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(df.corr(), annot=True, cmap='inferno')

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
print(scaled_data)

x = df.copy()
kmeans = KMeans(3)
kmeans.fit(x)
clusters = x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)

plt.figure(figsize=(10,10))
plt.scatter(clusters['Annual Income (k$)'],clusters['Spending Score (1-100)'],c=clusters['cluster_pred'],cmap='rainbow')
plt.title("Clustering customers based on Annual Income and Spending score", fontsize=15,fontweight="bold")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

SSE = []
for cluster in range(1,11):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(x)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,11), 'SSE':SSE})
plt.figure(figsize=(10, 10))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

print(silhouette_score(clusters, kmeans.labels_, metric='euclidean'))

kmeans_new = KMeans(5)
#Fit the data
kmeans_new.fit(x)
#Create a new data frame with the predicted clusters
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
#map the gender variable back to 'male' and 'female'
gender= {0:'Male',1:'Female'}
clusters_new['Gender']= clusters_new['Gender'].map(gender)

plt.figure(figsize=(10,10))
plt.scatter(clusters_new['Annual Income (k$)'],clusters_new['Spending Score (1-100)'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.title("Clustering customers based on Annual Income and Spending score", fontsize=15,fontweight="bold")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

avg_data = clusters_new.groupby(['cluster_pred'],
as_index=False).mean()
print(avg_data)

sns.barplot(x='cluster_pred',y='Age',palette="plasma",data=avg_data)
sns.barplot(x='cluster_pred',y='Annual Income (k$)',palette="plasma",data=avg_data)
sns.barplot(x='cluster_pred',y='Spending Score (1-100)',palette="plasma",data=avg_data)

data2 = pd.DataFrame(clusters_new.groupby(['cluster_pred','Gender'])['Gender'].count())
print(data2)