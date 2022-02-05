import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from seaborn import countplot

# Load dataset
df = pd.read_csv("/Users/nailamolooicloud.com/Downloads/Mall_Customers.csv")
print(df.head)
df.set_index('CustomerID')

plt.boxplot(df['Annual Income (k$)'])
pyplot.title("Boxplot of Annual Income")
pyplot.show()

sns.countplot(data['Gender'])
print(countplot.show())