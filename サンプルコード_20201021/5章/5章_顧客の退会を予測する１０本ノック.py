#!/usr/bin/env python
# coding: utf-8

# # 5章 顧客の退会を予測する１０本ノック
# 
# 引き続き、スポーツジムの会員データを使って顧客の行動を分析していきます。  
# ３章では顧客の全体像を把握し、4章では数ヶ月利用している顧客の来月の利用回数の予測を行いました。   
# ここでは、教師あり学習の分類を用いて、顧客の退会予測を取り扱います。

# In[5]:


import warnings
warnings.filterwarnings('ignore')


# ### ノック41：データを読み込んで利用データを整形しよう

# In[31]:


import pandas as pd
customer = pd.read_csv('customer_join.csv')
uselog_months = pd.read_csv('use_log_months.csv')
customer.head()


# In[11]:


# 当月+過去1ヶ月分の利用回数を集計
year_months = list(uselog_months["年月"].unique())
uselog = pd.DataFrame()

for i in range(1, len(year_months)):
    tmp = uselog_months.loc[uselog_months["年月"]==year_months[i]]
    tmp.rename(columns={"count":"count_0"}, inplace=True)
    tmp_before = uselog_months.loc[uselog_months["年月"]==year_months[i-1]]
    del tmp_before["年月"]
    tmp_before.rename(columns={"count":"count_1"}, inplace=True)
    tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    uselog = pd.concat([uselog, tmp], ignore_index=True)

uselog.head()


# In[ ]:





# In[ ]:





# ### ノック42：退会前月の退会顧客データを作成しよう

# In[40]:


from dateutil.relativedelta import relativedelta

# 退会したユーザー
exit_customer = customer.loc[customer["is_deleted"]==1]
exit_customer["exit_date"] = None
exit_customer["end_date"] = pd.to_datetime(exit_customer["end_date"])

# 退会前月(退会申請月)
for i in range(len(exit_customer)):
    exit_customer["exit_date"].iloc[i] = exit_customer["end_date"].iloc[i] - relativedelta(months=1)

exit_customer["exit_date"] = pd.to_datetime(exit_customer["exit_date"], format='%Y-%m-%d')
exit_customer["年月"] = exit_customer["exit_date"].dt.strftime("%Y%m")
uselog["年月"] = uselog["年月"].astype(str)

exit_uselog = pd.merge(uselog, exit_customer, on=["customer_id", "年月"], how="left")
print(len(uselog))
exit_uselog.head()


# In[41]:


# uselogベースで退会ユーザーのみ結合してるので欠損値が多い→消す
exit_uselog = exit_uselog.dropna(subset=["name"])
print(len(exit_uselog))
print(len(exit_uselog["customer_id"].unique()))
exit_uselog.head()


# ### ノック43：継続顧客のデータを作成しよう

# In[42]:


# 継続顧客のデータ
conti_customer = customer.loc[customer["is_deleted"]==0]
# uselogベースの結合(どの月のデータも使用できるのでidのみで結合)
conti_uselog = pd.merge(uselog, conti_customer, on=["customer_id"], how="left")
print(len(conti_uselog))
# 欠損値の除去
conti_uselog = conti_uselog.dropna(subset=["name"])
print(len(conti_uselog))


# In[45]:


# 顧客あたりデータが1件になるように調整(継続顧客が多すぎるため)
# アンダーサンプリング(データの不均衡時、多数側を減らして調整すること)

# 全体のデータをサンプリング(シャッフル)
conti_uselog = conti_uselog.sample(frac=1).reset_index(drop=True)
# 重複削除
conti_uselog = conti_uselog.drop_duplicates(subset="customer_id")
print(len(conti_uselog))
conti_uselog.head()


# In[46]:


# 継続顧客と退会顧客のデータを縦結合
predict_data = pd.concat([conti_uselog, exit_uselog],ignore_index=True)
print(len(predict_data))
predict_data.head()


# ### ノック44：予測する月の在籍期間を作成しよう

# In[50]:


predict_data["period"] = 0
predict_data["now_date"] = pd.to_datetime(predict_data["年月"], format="%Y%m")
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])

for i in range(len(predict_data)):
    delta = relativedelta(predict_data["now_date"][i], predict_data["start_date"][i])
    predict_data["period"][i] = int(delta.years*12 + delta.months)

predict_data.head()


# ### ノック45：欠損値を除去しよう

# In[48]:


predict_data.isna().sum()


# In[53]:


predict_data = predict_data.dropna(subset=['count_1'])
predict_data.isna().sum()


# ### ノック46：文字列型の変数を処理できるように整形しよう

# In[54]:


target_col = ["campaign_name", "class_name", "gender", "count_1", "routine_flg", "period", "is_deleted"]
predict_data = predict_data[target_col]
predict_data.head()


# In[55]:


# カテゴリカル変数→ダミー変数
predict_data = pd.get_dummies(predict_data)
predict_data.head()


# In[56]:


# gender_Fにフラグが立っていない = 男性 なので不要な列
del predict_data["gender_M"]
# 3項目の際も同じく、2列とも0なら残りの1つと判断できる
del predict_data["campaign_name_通常"]
del predict_data["class_name_ナイト"]

predict_data.head()


# ### ノック47：決定木を用いて退会予測モデルを作成してみよう

# In[60]:


# 決定木 直感的なのでよく最初に試される
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

exit = predict_data.loc[predict_data["is_deleted"]==1]
# 退会顧客と同数の継続顧客をランダムでサンプリング
conti = predict_data.loc[predict_data["is_deleted"]==0].sample(len(exit))

X = pd.concat([exit, conti], ignore_index=True)
y = X["is_deleted"]
del X["is_deleted"]
X_train, X_test, y_train, y_test = train_test_split(X,y)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
print(y_test_pred)


# In[61]:


results_test = pd.DataFrame({"y_test":y_test ,"y_pred":y_test_pred })
results_test.head()


# ### ノック48：予測モデルの評価を行ない、モデルのチューニングをしてみよう

# In[62]:


# 予測の正解件数
correct = len(results_test.loc[results_test["y_test"]==results_test["y_pred"]])
score_test = correct / len(results_test)
print(score_test)


# In[63]:


# 上記の予測を行える関数
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))


# In[64]:


# 過学習気味なので、チューニングの必要がある
# データを増やす、変数を見直す、モデルパラメータの変更等...

X = pd.concat([exit, conti], ignore_index=True)
y = X["is_deleted"]
del X["is_deleted"]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y)

# 決定木の深さの上限を5に設定
model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.score(X_train, y_train))


# ### ノック49：モデルに寄与している変数を確認しよう

# In[65]:


# 決定木における重要変数の取得
# 木構造を可視化することも可能
importance = pd.DataFrame({"feature_names":X.columns, "coefficient":model.feature_importances_})
importance


# ### ノック50：顧客の退会を予測しよう

# In[66]:


# 予測したいデータの作成
count_1 = 3
routine_flg = 1
period = 10
campaign_name = "入会費無料"
class_name = "オールタイム"
gender = "M"


# In[69]:


if campaign_name == "入会費半額":
    campaign_name_list = [1, 0]
elif campaign_name == "入会費無料":
    campaign_name_list = [0, 1]
elif campaign_name == "通常":
    campaign_name_list = [0, 0]
if class_name == "オールタイム":
    class_name_list = [1, 0]
elif class_name == "デイタイム":
    class_name_list = [0, 1]
elif class_name == "ナイト":
    class_name_list = [0, 0]
if gender == "F":
    gender_list = [1]
elif gender == "M":
    gender_list = [0]
input_data = [count_1, routine_flg, period]
input_data.extend(campaign_name_list)
input_data.extend(class_name_list)
input_data.extend(gender_list)
input_data


# In[72]:


# 退会予測
print(model.predict([input_data]))
# 退会確率予測(継続:退会)
print(model.predict_proba([input_data]))


# In[ ]:




