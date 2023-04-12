#!/usr/bin/env python
# coding: utf-8

# # 7章 ロジスティクスネットワークの最適設計を行う10本ノック
# 
# ここでは、最適化計算を行ういくつかのライブラリを用いて、最適化計算を実際に行っていきます。  
# そして、前章で用いたネットワーク可視化などの技術を駆使し、計算結果の妥当性を確認する方法についても学んでいきます。

# In[1]:


# 警告(worning)の非表示化
import warnings
warnings.filterwarnings('ignore')


# ### ノック６１：輸送最適化問題を解いてみよう

# In[2]:


# データ読み込み
import pandas as pd
df_tc = pd.read_csv('trans_cost.csv', index_col="工場")
df_tc


# In[3]:


df_demand = pd.read_csv('demand.csv')
df_demand


# In[4]:


df_supply = pd.read_csv('supply.csv')
df_supply


# In[19]:


import numpy as np
from itertools import product
from pulp import LpVariable, lpSum, value
from ortoolpy import model_min

# 初期設定 #
# 発生させる乱数の固定
np.random.seed(1)
nw = len(df_tc.index)
nf = len(df_tc.columns)
# product(): すべての組み合わせのタプル
pr = list(product(range(nw), range(nf)))

# 数理モデル作成 #
# 最小化を行うモデル
m1 = model_min()

# 目的関数の定義
# 変数(trans_costの各データを変数として扱う)
v1 = {(i,j): LpVariable('v%d_%d' %(i,j),lowBound=0) for i,j in pr}
# 変数と各要素の積の和を目的関数として定義
m1 += lpSum(df_tc.iloc[i][j] * v1[i,j] for i,j in pr)

print(v1)

# 制約条件の定義
# 供給
for i in range(nw):
    m1 += lpSum(v1[i,j] for j in range(nf)) <= df_supply.iloc[0][i]
# 需要
for j in range(nf):
    m1 += lpSum(v1[i,j] for i in range(nw)) >= df_demand.iloc[0][j]

# 最適化問題を解く
m1.solve()

# 総輸送コスト計算 #
df_tr_sol = df_tc.copy()
total_cost = 0

for k,x in v1.items():
    i,j = k[0],k[1]
    # value(x): 目的関数の値の取得
    df_tr_sol.iloc[i][j] = value(x)
    total_cost += df_tc.iloc[i][j] * value(x)

print(df_tr_sol)
print("総輸送コスト:" + str(total_cost))


# ### ノック６２：最適輸送ルートをネットワークで確認しよう

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# データ読み込み
df_tr = df_tr_sol.copy()
df_pos = pd.read_csv('trans_route_pos.csv')

# グラフオブジェクトの作成
G = nx.Graph()

# 頂点の設定
for i in range(len(df_pos.columns)):
    G.add_node(df_pos.columns[i])

# 辺の設定&エッジの重みのリスト化
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
pos = {}
for i in range(len(df_pos.columns)):
    node = df_pos.columns[i]
    pos[node] = (df_pos[node][0],df_pos[node][1])
    
# 描画
nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)

# 表示
plt.show()


# ### ノック６３：最適輸送ルートが制約条件内に収まっているかどうかを確認しよう

# In[7]:


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

print("需要条件計算結果:"+str(condition_demand(df_tr_sol,df_demand)))
print("供給条件計算結果:"+str(condition_supply(df_tr_sol,df_supply)))


# ### ノック６４：生産計画に関するデータを読み込んでみよう

# In[8]:


# 製品に対する必要な商品の割合
df_material = pd.read_csv('product_plan_material.csv', index_col="製品")
print(df_material)
# 製品の利益(売上高 - 売上原価)
df_profit = pd.read_csv('product_plan_profit.csv', index_col="製品")
print(df_profit)
# 原料の在庫数
df_stock = pd.read_csv('product_plan_stock.csv', index_col="項目")
print(df_stock)
# 各製品の生産量
df_plan = pd.read_csv('product_plan.csv', index_col="製品")
print(df_plan)


# ### ノック６５：利益を計算する関数を作ってみよう

# In[9]:


# 利益計算関数
def product_plan(df_profit, df_plan):
    profit = sum([df_profit.iloc[i][0] * df_plan.iloc[i][0] for i in range(len(df_profit.index))])
    return profit

print("総利益:" + str(product_plan(df_profit,df_plan)))


# ### ノック６６：生産最適化問題を解いてみよう

# In[16]:


import pandas as pd
from pulp import LpVariable, lpSum, value
from ortoolpy import model_max

# 必要材料数
df = df_material.copy()
# 材料の在庫
inv = df_stock
# 材料の種類数
ps = len(df_profit)

m = model_max()
# 変数,目的関数
v1 = {(i): LpVariable('v%d' %(i), lowBound=0) for i in range(ps)}
m += lpSum(df_profit.iloc[i] * v1[i] for i in range(ps))
# 制約条件
for i in range(len(df_material.columns)):
    m += lpSum(df_material.iloc[j,i] * v1[j] for j in range(ps)) <= df_stock.iloc[:,i]

m.solve()

df_plan_sol = df_plan.copy()

for k,x in v1.items():
    df_plan_sol.iloc[k] = value(x)

print(df_plan_sol)
# objective: 目的関数 にvalue()を使用して変数に代入している
print("総利益:" + str(value(m.objective)))


# ### ノック６７：最適生産計画が制約条件内に収まっているかどうかを確認しよう

# In[11]:


# 制約条件計算関数
def condition_stock(df_plan,df_material,df_stock):
    # 材料の種類数のゼロ配列
    flag = np.zeros(len(df_material.columns))
    
    for i in range(len(df_material.columns)):  
        temp_sum = 0
        # 製品数ループ
        for j in range(len(df_material.index)):
            # 材料の使用量
            temp_sum = temp_sum + df_material.iloc[j][i] * float(df_plan.iloc[j])
        if (temp_sum <= float(df_stock.iloc[0][i])):
            flag[i] = 1
        print(df_material.columns[i] + "  使用量:" + str(temp_sum)+", 在庫:" + str(float(df_stock.iloc[0][i])))
    return flag

print("制約条件計算結果:" + str(condition_stock(df_plan_sol,df_material,df_stock)))


# ### ノック６８：ロジスティクスネットワーク設計問題を解いてみよう

# In[12]:


import numpy as np
import pandas as pd
# 製品
prod = list('AB')
# 需要地(小売店)
dm = list('PQ')
# 工場
fc = list('XY')
# レーン
lane = (2,2)

# 輸送費表 #
tbdi = pd.DataFrame(((j,k) for j in dm for k in fc), columns=['需要地','工場'])
tbdi['輸送費'] = [1,2,3,1]
print(tbdi)

# 需要表 #
tbde = pd.DataFrame(((j,i) for j in dm for i in prod), columns=['需要地','製品'])
tbde['需要'] = [10,10,20,20]
print(tbde)

# 生産表 #
tbfa = pd.DataFrame(((f, l, p, 0, np.inf) for f in fc for l in range(len(lane)) for p in prod), 
                    columns=['工場','レーン','製品','下限','上限'])
tbfa['生産費'] = [1, np.nan, np.nan, 1, 3, np.nan, 5, 3]
tbfa.dropna(inplace=True)
tbfa.loc[4,'上限'] = 10
print(tbfa)

from ortoolpy import logistics_network
_, tbdi2, _ = logistics_network(tbde,tbdi,tbfa)

# ValY: 最適生産量
print(tbfa)
# ValX: 最適輸送量
print(tbdi2)


# ### ノック６９：最適ネットワークにおける輸送コストとその内訳を計算しよう

# In[13]:


print(tbdi2)
trans_cost = 0

for i in range(len(tbdi2.index)):
    trans_cost += tbdi2["輸送費"].iloc[i] * tbdi2["ValX"].iloc[i]

print("総輸送コスト:" + str(trans_cost))


# ### ノック７０：最適ネットワークにおける生産コストとその内訳を計算しよう

# In[14]:


print(tbfa)
product_cost = 0

for i in range(len(tbfa.index)):
    product_cost += tbfa["生産費"].iloc[i]*tbfa["ValY"].iloc[i]

print("総生産コスト:" + str(product_cost))


# In[ ]:




