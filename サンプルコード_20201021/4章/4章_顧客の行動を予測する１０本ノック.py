#!/usr/bin/env python
# coding: utf-8

# # 4章 顧客の全体像を把握する１０本ノック
# 
# 引き続き、スポーツジムの会員データを使って顧客の行動を分析していきます。  
# ３章で顧客の全体像を把握しました。  
# ここからは、機械学習を用いて顧客のグループ化や顧客の利用予測行なっていきましょう。  
# ここでは、教師なし学習、教師あり学習の回帰を取り扱います。

# ### ノック31：データを読み込んで確認しよう

# In[1]:


# 警告(worning)の非表示化
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
uselog = pd.read_csv('use_log.csv')
uselog.isnull().sum()


# In[2]:


customer = pd.read_csv('customer_join.csv')
customer.isnull().sum()


# ### ノック32：クラスタリングで顧客をグループ化しよう

# In[3]:


# 決められた正解がないので教師なし学習(クラスタリング)で予測する
# 利用履歴に基づいてグループ化を行う

customer_clustering = customer[["mean", "median","max", "min", "membership_period"]]
customer_clustering.head()


# In[4]:


from sklearn.cluster import KMeans # K-means法を使用
from sklearn.preprocessing import StandardScaler

# membership_period とそれ以外でデータの大きさが違うので標準化する
sc = StandardScaler()
customer_clustering_sc = sc.fit_transform(customer_clustering)

kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit(customer_clustering_sc)
customer_clustering["cluster"] = clusters.labels_
print(customer_clustering["cluster"].unique())
customer_clustering.head()


# ### ノック33：クラスタリング結果を分析しよう

# In[5]:


customer_clustering.columns = ["月内平均値","月内中央値", "月内最大値", "月内最小値","会員期間", "cluster"]
customer_clustering.groupby("cluster").count()


# In[6]:


customer_clustering.groupby("cluster").mean()


# ### ノック34：クラスタリング結果を可視化してみよう

# In[7]:


from sklearn.decomposition import PCA # 主成分分析

X = customer_clustering_sc
# 2次元に次元圧縮する
pca = PCA(n_components=2)
# 主成分の探索
pca.fit(X)
# 2つの主成分に対してデータを変換(次元削減)する
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df["cluster"] = customer_clustering["cluster"]


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 「5変数→2変数に次元削減しても分布が綺麗に分かれたままである」
# という事がわかる。残された2つの変数が何であるかはここでは深掘りしない。
for i in pca_df["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"]==i]
    plt.scatter(tmp[0], tmp[1])


# ### ノック35：クラスタリング結果をもとに退会顧客の傾向を把握しよう

# In[9]:


# customer_clusteringにis_deletedを追加
customer_clustering = pd.concat([customer_clustering, customer], axis=1)
customer_clustering_delflg = customer_clustering.groupby(["cluster","is_deleted"],as_index=False).count()[["cluster","is_deleted","customer_id"]]
customer_clustering_delflg.rename(columns={'customer_id':'count'}, inplace=True)
customer_clustering_delflg


# In[10]:


customer_clustering_rtnflg = customer_clustering.groupby(["cluster","routine_flg"],as_index=False).count()[["cluster","routine_flg","customer_id"]]
customer_clustering_rtnflg.rename(columns={'customer_id':'count'}, inplace=True)
customer_clustering_rtnflg


# ### ノック36：翌月の利用回数予測を行うためのデータ準備をしよう

# In[11]:


uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["年月"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["年月","customer_id"],as_index=False).count()
uselog_months.rename(columns={"log_id":"count"}, inplace=True)
del uselog_months["usedate"]
uselog_months.head()


# In[26]:


year_months = list(uselog_months["年月"].unique())
predict_data = pd.DataFrame()

for i in range(6, len(year_months)):
    tmp = uselog_months.loc[uselog_months["年月"]==year_months[i]]
    # 当月の利用回数(予測したい値)
    tmp.rename(columns={"count":"count_pred"}, inplace=True)
    
    for j in range(1, 7):
        # 当月からn(1-6)ヶ月前の利用回数
        tmp_before = uselog_months.loc[uselog_months["年月"]==year_months[i-j]]
        del tmp_before["年月"]
        tmp_before.rename(columns={"count":"count_{}month_before".format(j)}, inplace=True)
        tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    
    predict_data = pd.concat([predict_data, tmp], ignore_index=True)

predict_data.head()


# In[13]:


# 在籍期間が6ヶ月に満たない会員を除去
predict_data = predict_data.dropna()
predict_data = predict_data.reset_index(drop=True)
predict_data.head()


# ### ノック37：特徴となる変数を付与しよう

# In[14]:


predict_data = pd.merge(predict_data, customer[["customer_id","start_date"]], on="customer_id", how="left")
predict_data.head()


# In[16]:


# 月単位で会員期間を作成
predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])

from dateutil.relativedelta import relativedelta

predict_data["period"] = None

for i in range(len(predict_data)):
    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])
    predict_data["period"][i] = delta.years*12 + delta.months

predict_data.head()


# ### ノック38：来月の利用回数予測モデルを作成しよう

# In[19]:


# 2018/04以降に新規入会した顧客に絞ってモデル作成をする(利用回数が安定期に入っている可能性があるため)
predict_data = predict_data.loc[predict_data["start_date"]>=pd.to_datetime("20180401")]

from sklearn import linear_model # 線形回帰モデル
from sklearn.model_selection import train_test_split

model = linear_model.LinearRegression()
X = predict_data[["count_1month_before","count_2month_before","count_3month_before","count_4month_before","count_5month_before","count_6month_before","period"]]
y = predict_data["count_pred"]
X_train, X_test, y_train, y_test = train_test_split(X,y)
model.fit(X_train, y_train)


# In[20]:


print(model.score(X_train, y_train))
print(model.score(X_test, y_test))


# ### ノック39：モデルに寄与している変数を確認しよう

# In[21]:


# coefficient=係数
# 説明変数ごとにどれくらい予測に寄与しているかを確認する
coef = pd.DataFrame({"feature_names":X.columns, "coefficient":model.coef_})
coef


# ### ノック40：来月の利用回数を予測しよう

# In[27]:


x1 = [3, 4, 4, 6, 8, 7, 8]
x2 = [2, 2, 3, 3, 4, 6, 8]
x_pred = [x1, x2]


# In[28]:


model.predict(x_pred)


# In[31]:


uselog_months.to_csv("use_log_months.csv", index=False)


# In[ ]:




