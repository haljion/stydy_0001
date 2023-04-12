#!/usr/bin/env python
# coding: utf-8

# # 3章 顧客の全体像を把握する１０本ノック
# 
# ここでは、スポーツジムの会員データを使って顧客の行動を分析していきます。  
# これまでと同様にまずはデータを理解し、加工した後、  
# 顧客の行動データを分析していきましょう。  
# ここでは、機械学習に向けての初期分析を行います。

# ### ノック21：データを読み込んで把握しよう

# In[1]:


import pandas as pd
uselog = pd.read_csv('use_log.csv')
print(len(uselog))
uselog.head()


# In[2]:


customer = pd.read_csv('customer_master.csv')
print(len(customer))
customer.head()


# In[3]:


class_master = pd.read_csv('class_master.csv')
print(len(class_master))
class_master.head()


# In[4]:


campaign_master = pd.read_csv('campaign_master.csv')
print(len(campaign_master))
campaign_master.head()


# ### ノック22：顧客データを整形しよう

# In[5]:


customer_join = pd.merge(customer, class_master, on="class" ,how="left")
customer_join = pd.merge(customer_join, campaign_master, on="campaign_id" ,how="left")
customer_join.head()


# In[6]:


customer_join.isnull().sum()


# In[ ]:





# ### ノック23：顧客データの基礎集計をしよう

# In[7]:


customer_join.groupby("class_name").count()["customer_id"]


# In[8]:


customer_join.groupby("campaign_name").count()["customer_id"]


# In[9]:


customer_join.groupby("gender").count()["customer_id"]


# In[10]:


customer_join.groupby("is_deleted").count()["customer_id"]


# In[11]:


customer_join["start_date"] = pd.to_datetime(customer_join["start_date"])
customer_start = customer_join.loc[customer_join["start_date"]>pd.to_datetime("20180401")]
print(len(customer_start))


# ### ノック24：最新顧客データの基礎集計をしよう

# In[12]:


customer_join["end_date"] = pd.to_datetime(customer_join["end_date"])
customer_newer = customer_join.loc[(customer_join["end_date"]>=pd.to_datetime("20190331"))|(customer_join["end_date"].isna())]
print(len(customer_newer))
customer_newer["end_date"].unique()


# In[13]:


customer_newer.groupby("class_name").count()["customer_id"]


# In[14]:


customer_newer.groupby("campaign_name").count()["customer_id"]


# In[15]:


customer_newer.groupby("gender").count()["customer_id"]


# ### ノック25：利用履歴データを集計しよう

# In[16]:


uselog["usedate"] = pd.to_datetime(uselog["usedate"])
uselog["年月"] = uselog["usedate"].dt.strftime("%Y%m")
uselog_months = uselog.groupby(["年月","customer_id"],as_index=False).count()
uselog_months.rename(columns={"log_id":"count"}, inplace=True)
del uselog_months["usedate"]
uselog_months.head()


# In[27]:


uselog_customer = uselog_months.groupby("customer_id").agg(["mean", "median", "max", "min" ])["count"]
uselog_customer = uselog_customer.reset_index(drop=False)
uselog_customer.head()


# ### ノック26：利用履歴データから定期利用フラグを作成しよう

# In[18]:


uselog["weekday"] = uselog["usedate"].dt.weekday
uselog_weekday = uselog.groupby(["customer_id","年月","weekday"], as_index=False).count()
del uselog_weekday["usedate"]
uselog_weekday.rename(columns={"log_id":"count"}, inplace=True)
uselog_weekday.head()


# In[19]:


uselog_weekday = uselog_weekday.groupby("customer_id",as_index=False).max()[["customer_id", "count"]]
uselog_weekday["routine_flg"] = 0
uselog_weekday["routine_flg"] = uselog_weekday["routine_flg"].where(uselog_weekday["count"]<4, 1)
uselog_weekday.head()


# ### ノック27：顧客データと利用履歴データを結合しよう

# In[20]:


customer_join = pd.merge(customer_join, uselog_customer, on="customer_id", how="left")
customer_join = pd.merge(customer_join, uselog_weekday[["customer_id", "routine_flg"]], on="customer_id", how="left")
customer_join.head()


# In[21]:


customer_join.isnull().sum()


# ### ノック28：会員期間を計算しよう

# In[25]:


# 警告(worning)の非表示化
import warnings
warnings.filterwarnings('ignore')

from dateutil.relativedelta import relativedelta
customer_join["calc_date"] = customer_join["end_date"]
customer_join["calc_date"] = customer_join["calc_date"].fillna(pd.to_datetime("20190430"))
customer_join["membership_period"] = 0

for i in range(len(customer_join)):
    delta = relativedelta(customer_join["calc_date"].iloc[i], customer_join["start_date"].iloc[i])
    customer_join["membership_period"].iloc[i] = delta.years*12 + delta.months

customer_join.head()


# ### ノック29：顧客行動の各種統計量を把握しよう

# In[29]:


customer_join[['mean','median','max','min']].describe()


# In[31]:


customer_join.groupby("routine_flg").count()["customer_id"]


# In[32]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(customer_join['membership_period'])


# ### ノック30：退会ユーザーと継続ユーザーの違いを把握しよう

# In[37]:


customer_end = customer_join.loc[customer_join["is_deleted"]==1]
customer_end.describe()


# In[35]:


customer_stay = customer_join.loc[customer_join["is_deleted"]==0]
customer_stay.describe()


# In[38]:


customer_join.to_csv('customer_join.csv', index=False)


# In[ ]:




