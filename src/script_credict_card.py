# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
# %%
df = pd.read_csv('../database/CC_GENERAL.csv')
# %%
# data normalize, exclude id and tenure
df.drop(columns=['CUST_ID','TENURE'], inplace = True)

# analysis missing columns
missing = df.isna().sum()

# change data missing
df.fillna(df.median(), inplace = True)

#normalize data frame
df_normalize = Normalizer().fit_transform(df.values)
# %%
# clustering function
def clustering_algorithm(n_clusters, df):
  
  kmeans = KMeans(n_clusters, n_init = 10, max_iter = 300)
  labels = kmeans.fit_predict(df)
  labels_final = kmeans.labels_
  # calculate silhouette with data frame and labels, best values > 0 (positive)
  silhouette = metrics.silhouette_score(df, labels, metric = 'euclidean')
  
  # calculate index Davies Bouldin, best values next 0
  dbs = metrics.davies_bouldin_score(df, labels)
  
  # calculate Calinski
  calinski = metrics.calinski_harabasz_score(df, labels)
  
  print('Numero de clusters: ',n_clusters,silhouette, dbs, calinski)
  
  return silhouette, dbs, calinski, labels_final
# %%
# Make attempts with several cluster numbers, 
# considering the value of Silhouette as the main one, 
# the number that was better was with 5 cluster (Attempts = 3,5,10,20,50)
silhouette_final, dbs_final, calinski_final, labels_final = clustering_algorithm(5, df_normalize)
# %%
# create random data frame 
random_data = np.random.rand(8950, 16)
silhouette_random, dbs_random, calinski_random,labels_random = clustering_algorithm(5, random_data)
# %%
# Validating the stability of our dataset checking if
# splitting into 3 parts the results will be stable for each other
set1, set2, set3 = np.array_split(df_normalize, 3)
clustering_algorithm(5, set1)
clustering_algorithm(5, set2)
clustering_algorithm(5, set3)
# %%
# graphics
plt.scatter(df['PURCHASES'], df['PAYMENTS'], c = labels_final, s = 5, cmap = 'rainbow')
plt.xlabel('Valor total pago')
plt.ylabel('Valor total gasto')
plt.show
# %%
# show all possibility graphs about all columns
df['cluster'] = labels_final
sns.pairplot(df[0:], hue = 'cluster')
# %%
