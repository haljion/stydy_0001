#!/usr/bin/env python
# coding: utf-8

# # ２章　小売店のデータでデータ加工を行う１０本ノック
# 
# 本章では、ある小売店の売上履歴と顧客台帳データを用いて、データ分析の素地となる「データの加工」を習得することが目的です。
# 実際の現場データは手入力のExcel等、決して綺麗なデータではない事が多いため、
# データの揺れや整合性の担保など、汚いデータを取り扱うデータ加工を主体に進めて行きます。

# ### ノック１１：データを読み込んでみよう

# In[1]:


import pandas as pd
uriage_data = pd.read_csv("uriage.csv")
uriage_data.head()


# In[2]:


kokyaku_data = pd.read_excel("kokyaku_daicho.xlsx")
kokyaku_data.head()


# ### ノック１２：データの揺れを見てみよう

# In[3]:


uriage_data["item_name"].head()


# In[4]:


uriage_data["item_price"].head()


# In[ ]:





# ### ノック１３：データに揺れがあるまま集計しよう

# In[5]:


uriage_data["purchase_date"] = pd.to_datetime(uriage_data["purchase_date"])
uriage_data["purchase_month"] = uriage_data["purchase_date"].dt.strftime("%Y%m")
res = uriage_data.pivot_table(index="purchase_month", columns="item_name", aggfunc="size", fill_value=0)
res


# In[6]:


res = uriage_data.pivot_table(index="purchase_month", columns="item_name", values="item_price", aggfunc="sum", fill_value=0)
res


# ### ノック１４：商品名の揺れを補正しよう

# In[7]:


print(len(pd.unique(uriage_data.item_name)))


# In[8]:


uriage_data["item_name"] = uriage_data["item_name"].str.upper()
uriage_data["item_name"] = uriage_data["item_name"].str.replace("　", "")
uriage_data["item_name"] = uriage_data["item_name"].str.replace(" ", "")
uriage_data.sort_values(by=["item_name"], ascending=True)


# In[9]:


print(pd.unique(uriage_data.item_name))
print(len(pd.unique(uriage_data.item_name)))


# ### ノック１５：金額欠損値の補完をしよう

# In[10]:


uriage_data.isnull().any(axis=0)


# In[11]:


flg_is_null = uriage_data["item_price"].isnull()

for trg in list(uriage_data.loc[flg_is_null, "item_name"].unique()):
    price = uriage_data.loc[(~flg_is_null) & (uriage_data["item_name"] == trg), "item_price"].max()
    uriage_data["item_price"].loc[(flg_is_null) & (uriage_data["item_name"]==trg)] = price

uriage_data.head()


# In[12]:


uriage_data.isnull().any(axis=0)


# In[13]:


for trg in list(uriage_data["item_name"].sort_values().unique()):
    print(trg + "の最大額：" + str(uriage_data.loc[uriage_data["item_name"]==trg]["item_price"].max()) + "の最小額：" + str(uriage_data.loc[uriage_data["item_name"]==trg]["item_price"].min(skipna=False)))


# ### ノック１６：顧客名の揺れを補正しよう

# In[14]:


kokyaku_data["顧客名"].head()


# In[15]:


uriage_data["customer_name"].head()


# In[16]:


kokyaku_data["顧客名"] = kokyaku_data["顧客名"].str.replace(" ","")
kokyaku_data["顧客名"] = kokyaku_data["顧客名"].str.replace("　","")
kokyaku_data["顧客名"].head()


# ### ノック１７：日付の揺れを補正しよう

# In[19]:


flg_is_serial = kokyaku_data["登録日"].astype("str").str.isdigit()
flg_is_serial.sum()


# In[20]:


fromSerial = pd.to_timedelta(kokyaku_data.loc[flg_is_serial, "登録日"].astype("float"), unit="D") + pd.to_datetime("1900/01/01")
fromSerial


# In[21]:


fromString = pd.to_datetime(kokyaku_data.loc[~flg_is_serial, "登録日"])
fromString


# In[23]:


kokyaku_data["登録日"] = pd.concat([fromSerial,fromString])
kokyaku_data


# In[28]:


kokyaku_data["登録年月"] = kokyaku_data["登録日"].dt.strftime("%Y%m")
rslt = kokyaku_data.groupby("登録年月").count()["顧客名"]
print(rslt)
print(len(kokyaku_data))


# In[29]:


flg_is_serial = kokyaku_data["登録日"].astype("str").str.isdigit()
flg_is_serial.sum()


# ### ノック１８：顧客名をキーに２つのデータを結合(ジョイン)しよう

# In[31]:


join_data = pd.merge(uriage_data, kokyaku_data, left_on="customer_name", right_on="顧客名", how="left")
join_data = join_data.drop("customer_name", axis=1)
join_data


# 
# ### ノック１９：クレンジングしたデータをダンプしよう

# In[32]:


dump_data = join_data[["purchase_date", "purchase_month", "item_name", "item_price", "顧客名", "かな", "地域", "メールアドレス", "登録日"]]
dump_data


# In[33]:


dump_data.to_csv("dump_data.csv", index=False)


# ### ノック２０：データを集計しよう

# In[34]:


import_data = pd.read_csv("dump_data.csv")
import_data


# In[35]:


byItem = import_data.pivot_table(index="purchase_month", columns="item_name", aggfunc="size", fill_value=0)
byItem


# In[36]:


byPrice = import_data.pivot_table(index="purchase_month", columns="item_name", values="item_price", aggfunc="sum", fill_value=0)
byPrice


# In[37]:


byCustomer = import_data.pivot_table(index="purchase_month", columns="顧客名", aggfunc="size", fill_value=0)
byCustomer


# In[38]:


byRegion = import_data.pivot_table(index="purchase_month", columns="地域", aggfunc="size", fill_value=0)
byRegion


# In[39]:


away_data = pd.merge(uriage_data, kokyaku_data, left_on="customer_name", right_on="顧客名", how="right")
away_data[away_data["purchase_date"].isnull()][["顧客名", "メールアドレス", "登録日"]]


# In[ ]:




