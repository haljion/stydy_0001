#!/usr/bin/env python
# coding: utf-8

# # 6章 物流の最適ルートをコンサルティングする１０本ノック
# 
# ここでは、「物流」の基礎となる「輸送最適化」を検討するにあたっての基礎的な技術を習得します。  
# 実際の物流データからネットワーク構造を可視化する方法について学び、最適な物流計画を立案する流れを学んでいきます。

# In[39]:


# 警告(worning)の非表示化
import warnings
warnings.filterwarnings('ignore')


# ### ノック５１：物流に関するデータを読み込んでみよう

# In[1]:


import pandas as pd

# 工場データの読み込み
# demand: 要求数？
factories = pd.read_csv("tbl_factory.csv", index_col=0)
factories


# In[2]:


# 倉庫データの読み込み
warehouses = pd.read_csv("tbl_warehouse.csv", index_col=0)
warehouses


# In[6]:


# コストテーブル
cost = pd.read_csv("rel_cost.csv", index_col=0)
cost.head()


# In[4]:


# 輸送トランザクションテーブル
trans = pd.read_csv("tbl_transaction.csv", index_col=0)
trans.head()


# In[7]:


# トランザクションテーブルに各テーブルをジョインする
# コストデータを付与
join_data = pd.merge(trans, cost, left_on=["ToFC","FromWH"], right_on=["FCID","WHID"], how="left")
join_data.head()


# In[8]:


# 工場情報を付与
join_data = pd.merge(join_data, factories, left_on="ToFC", right_on="FCID", how="left")
join_data.head()


# In[9]:


# 倉庫情報を付与
join_data = pd.merge(join_data, warehouses, left_on="FromWH", right_on="WHID", how="left")
# カラムの並び替え
join_data = join_data[["TransactionDate","Quantity","Cost","ToFC","FCName","FCDemand","FromWH","WHName","WHSupply","WHRegion"]]
join_data.head()


# In[10]:


# 関東データを抽出
kanto = join_data.loc[join_data["WHRegion"]=="関東"]
kanto.head()


# In[11]:


# 東北データを抽出
tohoku = join_data.loc[join_data["WHRegion"]=="東北"]
tohoku.head()


# ### ノック５２：現状の輸送量、コストを確認してみよう

# In[12]:


# 支社のコスト合計を算出
print("関東支社の総コスト: " + str(kanto["Cost"].sum()) + "万円")
print("東北支社の総コスト: " + str(tohoku["Cost"].sum()) + "万円")


# In[13]:


# 支社の総輸送個数
print("関東支社の総部品輸送個数: " + str(kanto["Quantity"].sum()) + "個")
print("東北支社の総部品輸送個数: " + str(tohoku["Quantity"].sum()) + "個")


# In[14]:


# 部品一つ当たりの輸送コスト
tmp = (kanto["Cost"].sum() / kanto["Quantity"].sum()) * 10000
print("関東支社の部品１つ当たりの輸送コスト: " + str(int(tmp)) + "円")
tmp = (tohoku["Cost"].sum() / tohoku["Quantity"].sum()) * 10000
print("東北支社の部品１つ当たりの輸送コスト: " + str(int(tmp)) + "円")


# In[17]:


# コストテーブルを支社ごとに集計
cost_chk = pd.merge(cost, factories, on="FCID", how="left")
# 平均
print("東京支社の平均輸送コスト：" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"]=="関東"].mean()) + "万円")
print("東北支社の平均輸送コスト：" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"]=="東北"].mean()) + "万円")


# ### ノック５３：ネットワークを可視化してみよう

# In[20]:


import networkx as nx
import matplotlib.pyplot as plt

# グラフオブジェクトの作成
G=nx.Graph()

# 頂点の設定
G.add_node("nodeA")
G.add_node("nodeB")
G.add_node("nodeC")

# 辺の設定
G.add_edge("nodeA","nodeB")
G.add_edge("nodeA","nodeC")
G.add_edge("nodeB","nodeC")

# 座標の設定
pos={}
pos["nodeA"]=(0,0)
pos["nodeB"]=(1,1)
pos["nodeC"]=(0,1)

# 描画
nx.draw(G,pos)

# 表示
plt.show()


# ### ノック５４：ネットワークにノードを追加してみよう

# In[21]:


# 頂点Dを追加してみる
import networkx as nx
import matplotlib.pyplot as plt

# グラフオブジェクトの作成．
G=nx.Graph()

# 頂点の設定
G.add_node("nodeA")
G.add_node("nodeB")
G.add_node("nodeC")
G.add_node("nodeD")

# 辺の設定
G.add_edge("nodeA","nodeB")
G.add_edge("nodeA","nodeC")
G.add_edge("nodeB","nodeC")
G.add_edge("nodeA","nodeD")

# 座標の設定
pos={}
pos["nodeA"]=(0,0)
pos["nodeB"]=(1,1)
pos["nodeC"]=(0,1)
pos["nodeD"]=(1,0)

# 描画
nx.draw(G,pos, with_labels=True)

# 表示
plt.show()


# ### ノック５５：ルートの重みづけを実施しよう

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# データ読み込み
# リンクごとの重みを記載したファイル
df_w = pd.read_csv('network_weight.csv')
# 列数
df_w_cols = len(df_w.columns)

# 各リンクの位置を記載したファイル
df_p = pd.read_csv('network_pos.csv')

# エッジの重みのリスト化
size = 10
edge_weights = [df_w.iloc[i][j] * size for i in range(len(df_w)) for j in range(df_w_cols)]
        
# グラフオブジェクトの作成
G = nx.Graph()

# 頂点の設定
for i in range(df_w_cols):
    G.add_node(df_w.columns[i])

# 辺の設定
for i in range(df_w_cols):
    for j in range(df_w_cols):
        G.add_edge(df_w.columns[i],df_w.columns[j])

# 座標の設定
pos = {}
for i in range(df_w_cols):
    node = df_w.columns[i]
    pos[node] = (df_p[node][0],df_p[node][1])

# 描画
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 表示
plt.show()


# ### ノック５６：輸送ルート情報を読み込んでみよう

# In[59]:


import pandas as pd

# 輸送ルートごとの輸送量
df_tr = pd.read_csv('trans_route.csv', index_col="工場")
df_tr.head()


# ### ノック５７：輸送ルート情報からネットワークを可視化してみよう

# In[81]:


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# データ読み込み
df_tr = pd.read_csv('trans_route.csv', index_col="工場")
df_pos = pd.read_csv('trans_route_pos.csv')


# グラフオブジェクトの作成
G = nx.Graph()

# 頂点の設定
map(lambda p: G.add_node(p), df_pos.columns)
# for i in range(len(df_pos.columns)):
#     G.add_node(df_pos.columns[i])

# 辺の設定&エッジの重みのリスト化
# edge_weights = []
# size = 0.1

# for i in range(len(df_tr.index)):
#     for j in range(len(df_tr.columns)):
#         # 辺の追加
#         G.add_edge(df_tr.index[i],df_tr.columns[j])
#         edge_weights.append(df_tr.iloc[i,j] * size)

num_pre = 0
edge_weights = []
size = 0.1
for i in range(len(df_pos.columns)):
    for j in range(len(df_pos.columns)):
        if not (i==j):
            # 辺の追加
            G.add_edge(df_pos.columns[i],df_pos.columns[j])
            # エッジの重みの追加
            if num_pre<len(G.edges):
                num_pre = len(G.edges)
                weight = 0
                if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):
                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:
                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size
                elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):
                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:
                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size
                edge_weights.append(weight)        

# 座標の設定
pos = {c: (df_pos[c][0], df_pos[c][1]) for c in df_pos.columns}
# 描画
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 表示
plt.show()


# ### ノック５８：輸送コスト関数を作成しよう

# In[50]:


import pandas as pd

# データ読み込み
# 輸送ルート情報
df_tr = pd.read_csv('trans_route.csv', index_col="工場")
# 各ルートに必要なコスト
df_tc = pd.read_csv('trans_cost.csv', index_col="工場")

# 輸送コスト関数
def trans_cost(df_tr,df_tc):
    cost = 0
    for i in range(len(df_tc.index)):
        for j in range(len(df_tr.columns)):
            # 輸送実績 * 輸送コスト
            cost += df_tr.iloc[i][j] * df_tc.iloc[i][j]
    return cost

print("総輸送コスト:" + str(trans_cost(df_tr,df_tc)))


# ### ノック５９：制約条件を作ってみよう

# In[51]:


import pandas as pd

# データ読み込み
df_tr = pd.read_csv('trans_route.csv', index_col="工場")
# 工場の製品生産量に対する需要
df_demand = pd.read_csv('demand.csv')
# 倉庫が供給可能な部位品数の上限
df_supply = pd.read_csv('supply.csv')

# 需要側の制約条件
for i in range(len(df_demand.columns)):
    temp_sum = sum(df_tr[df_demand.columns[i]])
    print(str(df_demand.columns[i])+"への輸送量:"+str(temp_sum)+" (需要量:"+str(df_demand.iloc[0][i])+")")
    if temp_sum>=df_demand.iloc[0][i]:
        print("需要量を満たしています。")
    else:
        print("需要量を満たしていません。輸送ルートを再計算して下さい。")

# 供給側の制約条件
for i in range(len(df_supply.columns)):
    temp_sum = sum(df_tr.loc[df_supply.columns[i]])
    print(str(df_supply.columns[i])+"からの輸送量:"+str(temp_sum)+" (供給限界:"+str(df_supply.iloc[0][i])+")")
    if temp_sum<=df_supply.iloc[0][i]:
        print("供給限界の範囲内です。")
    else:
        print("供給限界を超過しています。輸送ルートを再計算して下さい。")


# ### ノック６０：輸送ルートを変更して、輸送コスト関数の変化を確認しよう

# In[52]:


import pandas as pd
import numpy as np

# データ読み込み
# 新しい輸送経路
df_tr_new = pd.read_csv('trans_route_new.csv', index_col="工場")
print(df_tr_new)

# 総輸送コスト再計算 
print("総輸送コスト(変更後):"+str(trans_cost(df_tr_new,df_tc)))

# 制約条件計算関数
# 需要側
def condition_demand(df_tr,df_demand):
    flag = np.zeros(len(df_demand.columns))
    for i in range(len(df_demand.columns)):
        temp_sum = sum(df_tr[df_demand.columns[i]])
        if (temp_sum>=df_demand.iloc[0][i]):
            flag[i] = 1
    return flag
            
# 供給側
def condition_supply(df_tr,df_supply):
    flag = np.zeros(len(df_supply.columns))
    for i in range(len(df_supply.columns)):
        temp_sum = sum(df_tr.loc[df_supply.columns[i]])
        if temp_sum<=df_supply.iloc[0][i]:
            flag[i] = 1
    return flag

print("需要条件計算結果:"+str(condition_demand(df_tr_new,df_demand)))
print("供給条件計算結果:"+str(condition_supply(df_tr_new,df_supply)))


# In[ ]:




