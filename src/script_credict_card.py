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
# generate cluster
kmeans = KMeans(n_clusters = 5, n_init = 10, max_iter = 300)
y_pred = kmeans.fit_predict(df_normalize)
# %%
labels = kmeans.labels_
# calculate silhouette with data frame and labels, best values > 0 (positive)
result_silhouette = metrics.silhouette_samples(df_normalize, labels, metric = 'euc')
# %%
# calculate index Davies Bouldin, best values next 0
dbs = metrics.davies_bouldin_score(df_normalize, labels)
# %%
# calculate Calinski
calinski = metrics.calinski_harabasz_score(df_normalize, labels)