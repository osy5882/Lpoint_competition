import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

df1 = pd.read_csv(r"C:\data\LPOINT_BIG_COMP_02_PDDE.csv")
df2 = pd.read_csv(r"C:\data\LPOINT_BIG_COMP_04_PD_CLAC.csv")
df_dom_clustered = pd.read_csv(r"C:\data\domain_cluster_result.csv")

df = pd.merge(df1, df2, how='left', on='pd_c')
# print(df)
# 여기까진 내가 가지고 있을 데이터 준비 완료


#함수
def merging(df1, df2):
    return pd.merge(df1, df2, how='left', on='cust')

def pivoting(df):
    df_pt = pd.pivot_table(data=df,
                           values='buy_am',
                           index='cust',
                           columns='clac_hlv_nm',
                           aggfunc='sum',
                           fill_value=0)

    scaler = MinMaxScaler()
    scaler.fit(df_pt)
    return pd.DataFrame(scaler.transform(df_pt), index=df_pt.index, columns=df_pt.columns)

def get_clusters_k(df):
    inertia_arr = []
    k_range = range(2, 20)
    rate_of_change = float('inf')
    k_result = None
    inertia_1 = float('inf')


    for k in k_range:
        Kmeans = KMeans(n_clusters=k, random_state=200)
        Kmeans.fit(df)
        inertia = Kmeans.inertia_
        if rate_of_change > (inertia_1 - inertia):
            rate_of_change = (inertia_1 - inertia)
            k_result = k-1
        inertia_1 = inertia
    return k_result

def clustering(df_main, df_clustered_1):
    df = merging(df_main, df_clustered_1)
    df_result = pd.DataFrame()
    for i in range(9):
        df_ = df[df['cluster']==i]
        df_pt = pivoting(df_)
        k = get_clusters_k(df_pt)
        Kmeans_ = KMeans(n_clusters=k, random_state=200)
        Kmeans_.fit(df_pt)
        cluster = Kmeans_.predict(df_pt)
        df_pt['buy_am_cluster'] = cluster
        df_result = pd.concat([df_result, df_pt])
        # break
    df_result.fillna(0, inplace=True)
    df_result.reset_index(drop=False)
    return df_result.merge(df_clustered_1, how="left", on="cust")

df_result = clustering(df, df_dom_clustered)
df_result.to_csv("../result.csv", index=False)
print(df_result)



