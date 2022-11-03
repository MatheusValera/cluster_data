# %%
import pandas as pd
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
print(df_normalize)
# %%
