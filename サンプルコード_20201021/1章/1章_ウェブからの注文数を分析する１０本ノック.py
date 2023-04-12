#!/usr/bin/env python
# coding: utf-8

# # １章 ウェブの注文数を分析する１０本ノック
# 
# ここでは、ある企業のECサイトでの商品の注文数の推移を分析していきます。  
# データの属性を理解し、分析をするためにデータを加工した後、  
# データの可視化を行うことで問題を発見していくプロセスを学びます。

# ### ノック１：データを読み込んでみよう

# In[3]:


import pandas as pd
customer_master = pd.read_csv('customer_master.csv')
customer_master.head() #先頭5行の表示


# In[5]:


item_master = pd.read_csv('item_master.csv')
item_master.head()


# In[6]:


transaction_1 = pd.read_csv('transaction_1.csv')
transaction_1.head()


# In[4]:


transaction_detail_1 = pd.read_csv('transaction_detail_1.csv')
transaction_detail_1.head()


# ### ノック２：データを結合(ユニオン)してみよう

# In[7]:


transaction_2 = pd.read_csv('transaction_2.csv')
transaction = pd.concat([transaction_1, transaction_2], ignore_index=True)
transaction.head()


# In[10]:


print(len(transaction_1))
print(len(transaction_2))
print(len(transaction))


# In[11]:


transaction_detail_2 = pd.read_csv('transaction_detail_2.csv')
transaction_detail = pd.concat([transaction_detail_1, transaction_detail_2], ignore_index=True)
transaction_detail.head()


# ### ノック３：売上データ同士を結合(ジョイン)してみよう

# In[12]:


join_data = pd.merge(transaction_detail, transaction[["transaction_id", "payment_date", "customer_id"]], on="transaction_id", how="left")
join_data.head()


# In[15]:


print(len(transaction_detail))
print(len(transaction))
print(len(join_data))


# ### ノック４：マスタデータを結合(ジョイン)してみよう

# In[17]:


join_data = pd.merge(join_data,customer_master, on="customer_id", how="left")
join_data = pd.merge(join_data,item_master, on="item_id", how="left")
join_data.head()


# ### ノック5：必要なデータ列を作ろう

# In[18]:


join_data["price"] = join_data["quantity"] * join_data["item_price"]
join_data.head()


# ### ノック6：データ検算をしよう

# In[19]:


print(join_data["price"].sum())
print(transaction["price"].sum())


# In[20]:


join_data["price"].sum() == transaction["price"].sum()


# ### ノック7：各種統計量を把握しよう

# In[22]:


join_data.isnull().sum()


# In[23]:


join_data.describe()


# In[24]:


print(join_data["payment_date"].min())
print(join_data["payment_date"].max())


# ### ノック8：月別でデータを集計してみよう

# In[26]:


join_data.dtypes


# In[27]:


join_data["payment_date"] = pd.to_datetime(join_data["payment_date"])
join_data["payment_month"] = join_data["payment_date"].dt.strftime("%Y%m")
join_data[["payment_date", "payment_month"]].head()


# In[29]:


join_data.groupby("payment_month").sum()["price"]


# ### ノック9：月別、商品別でデータを集計してみよう

# In[33]:


join_data.groupby(["payment_month","item_name"]).sum()[["price","quantity"]]


# In[35]:


pd.pivot_table(join_data,index="item_name",columns="payment_month",values=["price","quantity"],aggfunc="sum")


# ### ノック10：商品別の売上推移を可視化してみよう

# In[36]:


graph_data = pd.pivot_table(join_data,index="payment_month",columns="item_name",values="price",aggfunc="sum")
graph_data.head()


# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(list(graph_data.index), graph_data["PC-A"], label='PC-A')
plt.plot(list(graph_data.index), graph_data["PC-B"], label='PC-B')
plt.plot(list(graph_data.index), graph_data["PC-C"], label='PC-C')
plt.plot(list(graph_data.index), graph_data["PC-D"], label='PC-D')
plt.plot(list(graph_data.index), graph_data["PC-E"], label='PC-E')
plt.legend()


# In[39]:


graph_data.index


# In[ ]:




