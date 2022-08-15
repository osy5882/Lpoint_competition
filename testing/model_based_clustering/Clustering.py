import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# -- 1. 데이터 로드
df1 = pd.read_csv(r"C:\data\LPOINT_BIG_COMP_02_PDDE.csv")
df2 = pd.read_csv(r"C:\data\LPOINT_BIG_COMP_04_PD_CLAC.csv")
df_dom_clustered = pd.read_csv(r"C:\data\domain_cluster_result.csv")

df = pd.merge(df1, df2, how='left', on='pd_c')
# --> PD_CLAC (상품 정보) + PDDE(유통사 구매 기록) 테이블 결합

# print(df)
# 여기까진 내가 가지고 있을 데이터 준비 완료


# -- F1. merge (key = "cust")
def merging(df1, df2):
    return pd.merge(df1, df2, how='left', on='cust')

def make_scaler(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    return scaler

# -- F2. pivoting (--> 행: 고객, 열: 상품)
def pivoting(df):
    df_pt = pd.pivot_table(data=df,
                           values='buy_am',
                           index='cust',
                           columns='clac_hlv_nm',
                           aggfunc='sum',
                           fill_value=0)
    # ======== 문제 1 ======== (스케일링을 따로 빼줘야 할 거 같음) /// 수정 완료
    scaler = make_scaler(df_pt)
    return pd.DataFrame(scaler.transform(df_pt), index=df_pt.index, columns=df_pt.columns)


# -- F3. k를 얻는 과정 (로직 수정 여부 확인)
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

# -- F4. 클러스터링 과정
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



