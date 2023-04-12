#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# # 10章 アンケート分析を行うための言語処理１０本ノック
# 
# ここでは、まちづくりのアンケートを使って分析していきます。  
# 主に言語処理を取り扱っていきます。
# 言語処理特有の処理や、データの持たせ方を学びましょう。

# ### ノック91：データを読み込んで把握しよう

# In[2]:


import pandas as pd
survey = pd.read_csv("survey.csv")
print(len(survey))
survey.head()


# In[3]:


survey.isna().sum()


# In[4]:


survey = survey.dropna()
survey.isna().sum()


# ### ノック92：不要な文字を除外してみよう

# In[5]:


survey["comment"] = survey["comment"].str.replace("AA", "")
survey.head()


# In[6]:


# 正規表現
survey["comment"] = survey["comment"].str.replace("\(.+?\)", "", regex=True)
survey.head()


# In[7]:


survey["comment"] = survey["comment"].str.replace("\（.+?\）", "", regex=True)
survey.head()


# ### ノック93：文字数をカウントしてヒストグラムを表示してみよう

# In[8]:


# commentの文字数
survey["length"] = survey["comment"].str.len()
survey.head()


# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.hist(survey["length"])


# ### ノック94：形態素解析で文章を分割してみよう

# In[10]:


# 形態素解析(文章を単語,品詞に分割する技術)
# Mecab,Janome が代表的なライブラリ
import MeCab
# 初期化
tagger = MeCab.Tagger()
text = "すもももももももものうち"
# 形態素解析
words = tagger.parse(text)
words


# In[11]:


# \nで分割
words = tagger.parse(text).splitlines()
words_arr = []

for i in words:
    if i == 'EOS': continue
    # \tで分割した際の0番目(単語文字列)を取得
    word_tmp = i.split()[0]
    words_arr.append(word_tmp)

words_arr


# ### ノック95：形態素解析で文章から「動詞・名詞」を抽出してみよう

# In[12]:


text = "すもももももももものうち"
words = tagger.parse(text).splitlines()
words_arr = []
parts = ["名詞", "動詞"]
words = tagger.parse(text).splitlines()
words_arr = []
for i in words:
    if i == 'EOS' or i == '': continue
    word_tmp = i.split()[0]
    # 品詞
    part = i.split()[1].split(",")[0]
    if not (part in parts):continue
    words_arr.append(word_tmp)
words_arr


# ### ノック96：形態素解析で抽出した頻出する名詞を確認してみよう

# In[13]:


all_words = []
parts = ["名詞"]

# すべての名詞を抽出
for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = tagger.parse(text).splitlines()
    words_arr = []
    
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i.split()[0]
        part = i.split()[1].split(",")[0]
        if not (part in parts):continue
        words_arr.append(word_tmp)
    
    all_words.extend(words_arr)

print(all_words)


# In[14]:


# count列すべてに1を代入
all_words_df = pd.DataFrame({"words":all_words, "count":len(all_words)*[1]})
all_words_df = all_words_df.groupby("words").sum()
all_words_df.sort_values("count",ascending=False).head()


# ### ノック97：関係のない単語を除去しよう

# In[15]:


# 「の」はいらないので除去する
# 除外ワード
stop_words = ["の"]
all_words = []
parts = ["名詞"]

for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = tagger.parse(text).splitlines()
    words_arr = []
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i.split()[0]
        part = i.split()[1].split(",")[0]
        if not (part in parts):continue
        if word_tmp in stop_words:continue
        words_arr.append(word_tmp)
    all_words.extend(words_arr)

print(all_words)


# In[16]:


all_words_df = pd.DataFrame({"words":all_words, "count":len(all_words)*[1]})
all_words_df = all_words_df.groupby("words").sum()
all_words_df.sort_values("count",ascending=False).head()


# ### ノック98：顧客満足度と頻出単語の関係をみてみよう

# In[17]:


stop_words = ["の"]
parts = ["名詞"]
all_words = []
satisfaction = []

for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = tagger.parse(text).splitlines()
    words_arr = []
    
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i.split()[0]
        part = i.split()[1].split(",")[0]
        if not (part in parts):continue
        if word_tmp in stop_words:continue
        words_arr.append(word_tmp)
        satisfaction.append(survey["satisfaction"].iloc[n])
    
    all_words.extend(words_arr)

all_words_df = pd.DataFrame({"words":all_words, "satisfaction":satisfaction, "count":len(all_words)*[1]})
all_words_df.head()


# In[18]:


# 単語ごとの満足度の平均
words_satisfaction = all_words_df.groupby("words").mean()["satisfaction"]
words_count = all_words_df.groupby("words").sum()["count"]
# 横結合
words_df = pd.concat([words_satisfaction, words_count], axis=1)
words_df.head()


# In[19]:


# 特定のアンケート結果に引っ張られないよう、3回以上でてきた単語に限定
words_df = words_df.loc[words_df["count"]>=3]
# 降順
words_df.sort_values("satisfaction", ascending=False).head()


# In[20]:


# 昇順
words_df.sort_values("satisfaction").head()


# ### ノック99：アンケート毎の特徴を表現してみよう

# In[21]:


# どの単語が含まれているのかのみを特徴にする
parts = ["名詞"]
all_words_df = pd.DataFrame()
satisfaction = []

for n in range(len(survey)):
    text = survey["comment"].iloc[n]
    words = tagger.parse(text).splitlines()
    words_df = pd.DataFrame()
    
    for i in words:
        if i == "EOS" or i == "": continue
        word_tmp = i.split()[0]
        part = i.split()[1].split(",")[0]
        if not (part in parts):continue
        words_df[word_tmp] = [1]
    
    all_words_df = pd.concat([all_words_df, words_df] ,ignore_index=True)

all_words_df.head()


# In[22]:


all_words_df = all_words_df.fillna(0)
all_words_df.head()


# ### ノック100：類似アンケートを探してみよう

# In[23]:


print(survey["comment"].iloc[2])
target_text = all_words_df.iloc[2]
print(target_text)


# In[24]:


# コサイン類似度(特徴量同士の成す角度の近さで類似度を表す)
# 文書類似度の指標として代表的
import numpy as np
cos_sim = []

for i in range(len(all_words_df)):
    cos_text = all_words_df.iloc[i]
    cos = np.dot(target_text, cos_text) / (np.linalg.norm(target_text) * np.linalg.norm(cos_text))
    cos_sim.append(cos)

all_words_df["cos_sim"] = cos_sim
all_words_df.sort_values("cos_sim",ascending=False).head()


# In[25]:


print(survey["comment"].iloc[2])
print(survey["comment"].iloc[24])
print(survey["comment"].iloc[15])
print(survey["comment"].iloc[33])

