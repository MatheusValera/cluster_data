# %%
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn import metrics
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
def clustering_algorithm(n_clusters):
  
  kmeans = KMeans(n_clusters, n_init = 10, max_iter = 300)
  labels = kmeans.fit_predict(df_normalize)
  
  # calculate silhouette with data frame and labels, best values > 0 (positive)
  silhouette = metrics.silhouette_score(df_normalize, labels, metric = 'euclidean')
  
  # calculate index Davies Bouldin, best values next 0
  dbs = metrics.davies_bouldin_score(df_normalize, labels)
  
  # calculate Calinski
  calinski = metrics.calinski_harabasz_score(df_normalize, labels)
  
  print('Numero de clusters: ',n_clusters,silhouette, dbs, calinski)
  
  return silhouette, dbs, calinski
# %%
#Make attempts with several cluster numbers, 
# considering the value of Silhouette as the main one, 
# the number that was better was with 5 cluster (Attempts = 3,5,10,20,50)
clustering_algorithm(5)
# %%
